"""FastVLM processor adapted for the ViDoRe benchmark.

This file mirrors the behaviour of the FastVLM processor hosted in the user's
``colpali_engine`` fork so that the benchmark can be run without that external
package.
"""

from __future__ import annotations

from typing import ClassVar, Dict, List, Optional, Tuple, Union

import torch
from PIL import Image
from transformers import AutoTokenizer, BatchFeature, CLIPImageProcessor
from transformers.processing_utils import ProcessorMixin

# Constants used by the upstream FastVLM remote code.
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
IMAGE_TOKEN_INDEX = -200


def _score_multi_vector(
    qs: Union[torch.Tensor, List[torch.Tensor]],
    ps: Union[torch.Tensor, List[torch.Tensor]],
    *,
    batch_size: int = 128,
    device: Optional[Union[str, torch.device]] = None,
) -> torch.Tensor:
    """Compute MaxSim scores for multi-vector embeddings.

    The implementation follows the original ``colpali_engine`` helper.
    """

    if isinstance(qs, torch.Tensor):
        qs = list(torch.unbind(qs))
    if isinstance(ps, torch.Tensor):
        ps = list(torch.unbind(ps))

    if len(qs) == 0:
        raise ValueError("No queries provided")
    if len(ps) == 0:
        raise ValueError("No passages provided")

    device = device or (qs[0].device if isinstance(qs, list) else torch.device("cpu"))
    device = torch.device(device)

    scores_list: List[torch.Tensor] = []

    for i in range(0, len(qs), batch_size):
        scores_batch = []
        qs_batch = torch.nn.utils.rnn.pad_sequence(qs[i : i + batch_size], batch_first=True, padding_value=0).to(device)
        for j in range(0, len(ps), batch_size):
            ps_batch = torch.nn.utils.rnn.pad_sequence(ps[j : j + batch_size], batch_first=True, padding_value=0).to(device)
            scores_batch.append(torch.einsum("bnd,csd->bcns", qs_batch, ps_batch).max(dim=3)[0].sum(dim=2))
        scores_batch = torch.cat(scores_batch, dim=1).cpu()
        scores_list.append(scores_batch)

    scores = torch.cat(scores_list, dim=0)
    assert scores.shape[0] == len(qs), f"Expected {len(qs)} scores, got {scores.shape[0]}"
    return scores.to(torch.float32)


class ColFastVLMProcessor(ProcessorMixin):
    """Processor for FastVLM models producing ColBERT-style embeddings."""

    visual_prompt_prefix: ClassVar[str] = "Describe the image in detail."
    query_prefix: ClassVar[str] = "Query: "
    query_augmentation_token: ClassVar[str] = " "
    image_token: ClassVar[str] = DEFAULT_IMAGE_TOKEN
    im_start_token: ClassVar[str] = DEFAULT_IM_START_TOKEN
    im_end_token: ClassVar[str] = DEFAULT_IM_END_TOKEN
    fused_visual_prompt_template: ClassVar[str] = "{im_start}{image}{im_end}Describe the image."

    def __init__(
        self,
        pretrained_model_name_or_path: str = "apple/FastVLM-0.5B",
        use_fused_prompt: bool = True,
        **kwargs,
    ) -> None:
        tokenizer_kwargs: Dict = dict(kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, **tokenizer_kwargs)

        self.image_processor = CLIPImageProcessor(
            crop_size={"height": 1024, "width": 1024},
            image_mean=[0.0, 0.0, 0.0],
            image_std=[1.0, 1.0, 1.0],
            size={"shortest_edge": 1024},
            do_normalize=True,
            do_resize=True,
            do_center_crop=True,
            resample=3,
        )

        self.processor = type(
            "FastVLMProcessor", (), {"tokenizer": self.tokenizer, "image_processor": self.image_processor}
        )()

        if hasattr(self.tokenizer, "padding_side"):
            self.tokenizer.padding_side = "left"

        self.use_fused_prompt = use_fused_prompt
        if hasattr(self.tokenizer, "add_tokens"):
            needed = [
                token
                for token in [self.image_token, self.im_start_token, self.im_end_token]
                if token not in self.tokenizer.get_vocab()
            ]
            if needed:
                self.tokenizer.add_tokens(needed, special_tokens=True)

    @property
    def image_token_id(self) -> int:
        return self.tokenizer.convert_tokens_to_ids(self.image_token)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        return cls(pretrained_model_name_or_path=pretrained_model_name_or_path, **kwargs)

    def save_pretrained(self, save_directory: str, **kwargs):
        self.tokenizer.save_pretrained(save_directory, **kwargs)

    def process_images(
        self,
        images: List[Image.Image],
        context_prompts: Optional[List[str]] = None,
    ) -> BatchFeature:
        if context_prompts and len(images) != len(context_prompts):
            raise ValueError("Length of images and context prompts must match.")

        if context_prompts:
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

        images = [image.convert("RGB") for image in images]

        text_inputs = self.tokenizer(
            texts_doc,
            return_tensors="pt",
            padding="longest",
        )

        original_input_ids = text_inputs["input_ids"].clone()

        if self.use_fused_prompt:
            image_token_id = self.image_token_id
            if image_token_id == self.tokenizer.unk_token_id:
                raise ValueError(
                    "FastVLM tokenizer does not recognize the image token. "
                    "Ensure additional special tokens are registered."
                )
            input_ids = text_inputs["input_ids"].clone()
            input_ids[input_ids == image_token_id] = IMAGE_TOKEN_INDEX
            text_inputs["input_ids"] = input_ids

        text_inputs["text_input_ids"] = original_input_ids

        image_inputs = self.image_processor(images, return_tensors="pt")

        batch_doc = BatchFeature({**text_inputs, "images": image_inputs.get("pixel_values")})
        return batch_doc

    def process_queries(
        self,
        queries: List[str],
        max_length: int = 50,
        suffix: Optional[str] = None,
    ) -> BatchFeature:
        if suffix is None:
            suffix = self.query_augmentation_token * 10

        texts_query: List[str] = []
        for query in queries:
            texts_query.append(self.query_prefix + query + suffix)

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
        *,
        device: Optional[Union[str, torch.device]] = None,
        batch_size: int = 128,
        **kwargs,
    ) -> torch.Tensor:
        return _score_multi_vector(qs, ps, batch_size=batch_size, device=device)

    def get_n_patches(
        self,
        image_size: Tuple[int, int],
        patch_size: int = 64,
    ) -> Tuple[int, int]:
        default_image_size = 1024

        if hasattr(self.image_processor, "size"):
            if isinstance(self.image_processor.size, dict):
                if "height" in self.image_processor.size:
                    default_image_size = self.image_processor.size["height"]
                elif "shortest_edge" in self.image_processor.size:
                    default_image_size = self.image_processor.size["shortest_edge"]
            else:
                default_image_size = self.image_processor.size

        n_patches_h = default_image_size // patch_size
        n_patches_w = default_image_size // patch_size

        return n_patches_h, n_patches_w

    def get_image_mask(self, batch_images: BatchFeature) -> torch.Tensor:
        if "input_ids" in batch_images:
            return torch.zeros_like(batch_images.input_ids, dtype=torch.bool)
        return torch.tensor([], dtype=torch.bool)

    def __call__(self, text=None, images=None, return_tensors="pt", padding="longest", **kwargs):
        if images is not None:
            images = [img.convert("RGB") if hasattr(img, "convert") else img for img in images]
            image_inputs = self.image_processor(images, return_tensors=return_tensors, **kwargs)
        else:
            image_inputs = {}

        if text is not None:
            text_inputs = self.tokenizer(text, return_tensors=return_tensors, padding=padding, **kwargs)
        else:
            text_inputs = {}

        combined_inputs: Dict = {}
        combined_inputs.update(text_inputs)
        if images is not None:
            combined_inputs["images"] = image_inputs.get("pixel_values")

        return BatchFeature(combined_inputs)
