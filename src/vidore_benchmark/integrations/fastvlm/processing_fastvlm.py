from typing import ClassVar, List, Optional, Tuple, Union

import torch
from PIL import Image
from transformers import AutoTokenizer, BatchFeature, CLIPImageProcessor
from transformers.processing_utils import ProcessorMixin

from .processing_utils import BaseVisualRetrieverProcessor

from .llava_qwen import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX


class ColFastVLMProcessor(BaseVisualRetrieverProcessor, ProcessorMixin):
    """
    Processor for ColFastVLM models.

    This processor handles both image and text inputs for FastVLM models
    in the context of ColPali-style document retrieval.

    Args:
        pretrained_model_name_or_path: Path or name of the pretrained FastVLM model
        **kwargs: Additional arguments passed to the base processor
    """

    # Default simple prompt (manual path)
    visual_prompt_prefix: ClassVar[str] = "Describe the image in detail."
    query_prefix: ClassVar[str] = "Query: "
    query_augmentation_token: ClassVar[str] = " "  # Simple space for padding
    # FastVLM (LLaVA-Qwen style) special tokens
    image_token: ClassVar[str] = DEFAULT_IMAGE_TOKEN
    im_start_token: ClassVar[str] = DEFAULT_IM_START_TOKEN
    im_end_token: ClassVar[str] = DEFAULT_IM_END_TOKEN
    # Fused visual prompt pattern
    fused_visual_prompt_template: ClassVar[str] = "{im_start}{image}{im_end}Describe the image."

    def __init__(
        self,
        pretrained_model_name_or_path: str = "apple/FastVLM-0.5B",
        use_fused_prompt: bool = True,
        **kwargs,
    ):
        # FastVLM doesn't have a unified processor, so we need to load components separately

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)

        # Create image processor manually with FastVLM/MobileCLIP parameters
        # These parameters match exactly what FastVLM's vision tower expects
        self.image_processor = CLIPImageProcessor(
            crop_size={"height": 1024, "width": 1024},
            image_mean=[0.0, 0.0, 0.0],  # FastVLM/MobileCLIP normalization
            image_std=[1.0, 1.0, 1.0],
            size={"shortest_edge": 1024},
            do_normalize=True,
            do_resize=True,
            do_center_crop=True,
            resample=3,  # PIL.Image.BICUBIC, matches FastVLM
        )

        # For compatibility, create a simple processor object
        self.processor = type(
            "FastVLMProcessor", (), {"tokenizer": self.tokenizer, "image_processor": self.image_processor}
        )()

        # Set tokenizer padding side
        if hasattr(self.tokenizer, "padding_side"):
            self.tokenizer.padding_side = "left"

        self.use_fused_prompt = use_fused_prompt
        if hasattr(self.tokenizer, "add_tokens"):
            needed = [
                t
                for t in [self.image_token, self.im_start_token, self.im_end_token]
                if t not in self.tokenizer.get_vocab()
            ]
            if needed:
                self.tokenizer.add_tokens(needed, special_tokens=True)

    @property
    def image_token_id(self) -> int:
        return self.tokenizer.convert_tokens_to_ids(self.image_token)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        """
        Load a pretrained processor.

        Args:
            pretrained_model_name_or_path: Path or name of the pretrained model
            **kwargs: Additional arguments

        Returns:
            ColFastVLMProcessor: Loaded processor instance
        """
        return cls(pretrained_model_name_or_path=pretrained_model_name_or_path, **kwargs)

    def save_pretrained(self, save_directory: str, **kwargs):
        """
        Save the processor to a directory.

        Args:
            save_directory: Directory to save the processor to
            **kwargs: Additional arguments
        """
        # Save tokenizer and image processor separately
        self.tokenizer.save_pretrained(save_directory, **kwargs)
        # Note: CLIPImageProcessor doesn't have save_pretrained, so we skip it
        # The image processor config is recreated in __init__ based on FastVLM specs

    def process_images(
        self,
        images: List[Image.Image],
        context_prompts: Optional[List[str]] = None,
    ) -> BatchFeature:
        """
        Process images for ColFastVLM.

        Args:
            images: List of PIL images to process
            context_prompts: Optional list of context prompts for each image

        Returns:
            BatchFeature: Processed batch of images and text
        """
        if context_prompts:
            if len(images) != len(context_prompts):
                raise ValueError("Length of images and context prompts must match.")
            texts_doc = context_prompts
        else:
            if self.use_fused_prompt:
                fused_prompt = self.fused_visual_prompt_template.format(
                    im_start=self.im_start_token,
                    image=self.image_token,
                    im_end=self.im_end_token,
                )
                texts_doc = [fused_prompt] * len(images)
            else:
                texts_doc = [self.visual_prompt_prefix] * len(images)

        # Convert images to RGB
        images = [image.convert("RGB") for image in images]

        # Process with our custom FastVLM processor components
        # Process text
        text_inputs = self.tokenizer(
            texts_doc,
            return_tensors="pt",
            padding="longest",
        )

        original_input_ids = text_inputs["input_ids"].clone()

        if self.use_fused_prompt:
            image_token_id = self.tokenizer.convert_tokens_to_ids(self.image_token)
            if image_token_id == self.tokenizer.unk_token_id:
                raise ValueError(
                    "FastVLM tokenizer does not recognize the image token. "
                    "Ensure additional special tokens are registered."
                )
            input_ids = text_inputs["input_ids"].clone()
            input_ids[input_ids == image_token_id] = IMAGE_TOKEN_INDEX
            text_inputs["input_ids"] = input_ids

        text_inputs["text_input_ids"] = original_input_ids

        # Process images
        image_inputs = self.image_processor(images, return_tensors="pt")

        # Combine into FastVLM format
        batch_doc = BatchFeature({**text_inputs, "images": image_inputs.get("pixel_values")})

        return batch_doc

    def process_queries(
        self,
        queries: List[str],
        max_length: int = 50,
        suffix: Optional[str] = None,
    ) -> BatchFeature:
        """
        Process text queries for ColFastVLM.

        Args:
            queries: List of query strings
            max_length: Maximum length for tokenization (not strictly enforced in FastVLM)
            suffix: Optional suffix to add to queries for augmentation

        Returns:
            BatchFeature: Processed batch of queries
        """
        if suffix is None:
            # Use padding tokens as suffix for query augmentation
            suffix = self.query_augmentation_token * 10

        texts_query: List[str] = []
        for query in queries:
            # Add query prefix and suffix
            query = self.query_prefix + query + suffix
            texts_query.append(query)

        # Process queries with tokenizer
        batch_query = self.tokenizer(
            texts_query,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=max_length if max_length else self.tokenizer.model_max_length,
        )

        return batch_query

    def score(
        self,
        qs: List[torch.Tensor],
        ps: List[torch.Tensor],
        device: Optional[Union[str, torch.device]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute the MaxSim score (ColBERT-like) for the given multi-vector query and passage embeddings.

        Args:
            qs: List of query embedding tensors
            ps: List of passage (document) embedding tensors
            device: Device to use for computation
            **kwargs: Additional arguments

        Returns:
            torch.Tensor: Score matrix of shape (n_queries, n_passages)
        """
        return self.score_multi_vector(qs, ps, device=device, **kwargs)

    def get_n_patches(
        self,
        image_size: Tuple[int, int],
        patch_size: int = 64,
    ) -> Tuple[int, int]:
        """
        Get the number of patches for an image given the patch size.

        FastVLM uses MobileCLIP with a patch size of 64x64 by default.

        Args:
            image_size: Tuple of (height, width) for the image
            patch_size: Size of each patch (default 64 for FastVLM)

        Returns:
            Tuple[int, int]: Number of patches in (height, width)
        """
        # FastVLM/MobileCLIP uses 1024x1024 images by default with 64x64 patches
        # This gives 16x16 = 256 patches
        default_image_size = 1024

        # If the image processor has specific size settings, use those
        if hasattr(self.image_processor, "size"):
            if isinstance(self.image_processor.size, dict):
                if "height" in self.image_processor.size:
                    default_image_size = self.image_processor.size["height"]
                elif "shortest_edge" in self.image_processor.size:
                    default_image_size = self.image_processor.size["shortest_edge"]
            else:
                default_image_size = self.image_processor.size

        # Calculate number of patches
        n_patches_h = default_image_size // patch_size
        n_patches_w = default_image_size // patch_size

        return n_patches_h, n_patches_w

    def get_image_mask(self, batch_images: BatchFeature) -> torch.Tensor:
        """
        Get a mask indicating image token positions.

        Note: FastVLM doesn't use special image tokens based on the config,
        so this method returns a dummy mask. This is kept for API compatibility.

        Args:
            batch_images: Batch of processed images

        Returns:
            torch.Tensor: Boolean mask (all False for FastVLM)
        """
        # FastVLM doesn't use special image tokens
        # Return a mask of all False for compatibility
        if "input_ids" in batch_images:
            return torch.zeros_like(batch_images.input_ids, dtype=torch.bool)
        else:
            # Return empty tensor if no input_ids
            return torch.tensor([], dtype=torch.bool)

    def __call__(self, text=None, images=None, return_tensors="pt", padding="longest", **kwargs):
        """
        Call the underlying processor components.

        This method handles multimodal inputs by processing text and images separately
        and combining them into a FastVLM-compatible format.
        """
        # Handle image processing
        if images is not None:
            images = [img.convert("RGB") if hasattr(img, "convert") else img for img in images]
            image_inputs = self.image_processor(images, return_tensors=return_tensors, **kwargs)
        else:
            image_inputs = {}

        # Handle text processing
        if text is not None:
            text_inputs = self.tokenizer(text, return_tensors=return_tensors, padding=padding, **kwargs)
        else:
            text_inputs = {}

        # Combine inputs for FastVLM format
        combined_inputs = {}
        combined_inputs.update(text_inputs)
        if images is not None:
            combined_inputs["images"] = image_inputs.get("pixel_values")

        return BatchFeature(combined_inputs)
