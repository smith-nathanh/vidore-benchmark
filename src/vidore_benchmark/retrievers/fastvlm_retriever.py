from __future__ import annotations

import json
import logging
import os
from typing import Dict, List, Optional, Union, cast

import torch
from dotenv import load_dotenv
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers.utils.import_utils import is_flash_attn_2_available

from vidore_benchmark.integrations.fastvlm import ColFastVLM, ColFastVLMProcessor
from vidore_benchmark.retrievers.base_vision_retriever import BaseVisionRetriever
from vidore_benchmark.retrievers.registry_utils import register_vision_retriever
from vidore_benchmark.utils.data_utils import ListDataset
from vidore_benchmark.utils.torch_utils import get_torch_device

logger = logging.getLogger(__name__)

load_dotenv(override=True)


@register_vision_retriever("fastvlm")
class FastVLMRetriever(BaseVisionRetriever):
    """FastVLM retriever backed by the benchmark's vendored wrappers."""

    def __init__(
        self,
        pretrained_model_name_or_path: str = "apple/FastVLM-0.5B",
        device: str = "auto",
        num_workers: int = 0,
        mask_non_image_embeddings: bool = True,  # Match training config
        fuse_in_decoder: bool = True,
        use_fused_prompt: bool = True,
        torch_dtype: Optional[torch.dtype] = torch.bfloat16,
        processor_kwargs: Optional[Dict] = None,
        **kwargs,
    ) -> None:
        super().__init__(use_visual_embedding=True)

        self.device = get_torch_device(device)
        self.num_workers = num_workers

        attn_implementation = "flash_attention_2" if is_flash_attn_2_available() else None

        model_kwargs = {
            "trust_remote_code": True,
            "mask_non_image_embeddings": mask_non_image_embeddings,
            "fuse_in_decoder": fuse_in_decoder,
            "device_map": self.device,
        }
        if torch_dtype is not None:
            model_kwargs["torch_dtype"] = torch_dtype
        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation
        model_kwargs.update(kwargs)

        self.model = cast(
            ColFastVLM,
            ColFastVLM.from_pretrained(
                pretrained_model_name_or_path,
                **model_kwargs,
            ).eval(),
        )

        processor_extra_kwargs: Dict = {
            "trust_remote_code": True,
            "use_fused_prompt": use_fused_prompt,
        }
        if processor_kwargs:
            processor_extra_kwargs.update(processor_kwargs)

        # For PEFT checkpoints, load processor from base model
        processor_path = pretrained_model_name_or_path
        adapter_config_path = os.path.join(pretrained_model_name_or_path, "adapter_config.json")
        if os.path.exists(adapter_config_path):
            # This is a PEFT checkpoint, use base model for processor
            with open(adapter_config_path, 'r') as f:
                adapter_config = json.load(f)
            processor_path = adapter_config.get("base_model_name_or_path", "apple/FastVLM-0.5B")
            print(f"Using processor from base model: {processor_path}")

        self.processor = cast(
            ColFastVLMProcessor,
            ColFastVLMProcessor.from_pretrained(
                processor_path,
                **processor_extra_kwargs,
            ),
        )

    def process_images(self, images: List[Image.Image], **kwargs):
        return self.processor.process_images(images=images).to(self.device)

    def process_queries(self, queries: List[str], **kwargs):
        return self.processor.process_queries(queries=queries).to(self.device)

    def forward_queries(self, queries: List[str], batch_size: int, **kwargs) -> List[torch.Tensor]:
        dataloader = DataLoader(
            dataset=ListDataset[str](queries),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self.process_queries,
            num_workers=self.num_workers,
        )

        query_embeddings: List[torch.Tensor] = []

        with torch.no_grad():
            for batch_query in tqdm(dataloader, desc="Forward pass queries...", leave=False):
                embeddings_query = self.model(**batch_query).to("cpu")
                query_embeddings.extend(list(torch.unbind(embeddings_query)))

        return query_embeddings

    def forward_passages(self, passages: List[Image.Image], batch_size: int, **kwargs) -> List[torch.Tensor]:
        dataloader = DataLoader(
            dataset=ListDataset[Image.Image](passages),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self.process_images,
            num_workers=self.num_workers,
        )

        passage_embeddings: List[torch.Tensor] = []

        with torch.no_grad():
            for batch_doc in tqdm(dataloader, desc="Forward pass documents...", leave=False):
                embeddings_doc = self.model(**batch_doc).to("cpu")
                passage_embeddings.extend(list(torch.unbind(embeddings_doc)))

        return passage_embeddings

    def get_scores(
        self,
        query_embeddings: Union[torch.Tensor, List[torch.Tensor]],
        passage_embeddings: Union[torch.Tensor, List[torch.Tensor]],
        batch_size: Optional[int] = 128,
    ) -> torch.Tensor:
        if batch_size is None:
            raise ValueError("`batch_size` must be provided for FastVLMRetriever's scoring")
        scores = self.processor.score(
            qs=query_embeddings,
            ps=passage_embeddings,
            batch_size=batch_size,
            device="cpu",
        )
        return scores
