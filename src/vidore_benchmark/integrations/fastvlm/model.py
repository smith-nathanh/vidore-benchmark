"""ColFastVLM modeling wrapper adapted for the ViDoRe benchmark.

This module is derived from the user's FastVLM extension of ``colpali_engine`` and
removes the dependency on that library so that FastVLM can be consumed directly
from the benchmark package.
"""

from __future__ import annotations

import os
import json
import warnings
from types import MethodType
from typing import ClassVar, Optional, Dict, Any

import torch
from torch import nn
from transformers import AutoModelForCausalLM

# Sentinel value used by the upstream FastVLM remote code when injecting image tokens.
IMAGE_TOKEN_INDEX: ClassVar[int] = -200


def _load_model_with_peft_support(
    pretrained_model_name_or_path: str,
    **kwargs
) -> tuple[Any, Dict[str, torch.Tensor]]:
    """
    Load a model, automatically detecting if it's a PEFT checkpoint.
    
    Returns:
        tuple: (model, custom_text_proj_state_dict)
            - model: The loaded model (either base or with PEFT adapter merged)
            - custom_text_proj_state_dict: State dict for custom_text_proj layer if found, else empty dict
    """
    custom_text_proj_state = {}
    model_kwargs = dict(kwargs)
    model_kwargs.setdefault("trust_remote_code", True)
    
    # Check if this is a PEFT checkpoint
    adapter_config_path = os.path.join(pretrained_model_name_or_path, "adapter_config.json")
    is_peft_checkpoint = os.path.isfile(adapter_config_path)
    
    if is_peft_checkpoint:
        try:
            from peft import PeftModel
            from safetensors import safe_open
        except ImportError:
            raise ImportError(
                "PEFT and safetensors are required to load LoRA checkpoints. "
                "Install with: pip install peft safetensors"
            )
        
        # Read adapter config to find base model
        with open(adapter_config_path, 'r') as f:
            adapter_config = json.load(f)
        
        # Get base model name, fallback to FastVLM if not specified
        base_model_name = adapter_config.get("base_model_name_or_path")
        if not base_model_name:
            # Default to FastVLM base model if not specified
            base_model_name = "apple/FastVLM-0.5B"
            print(f"No base model specified in adapter config, using default: {base_model_name}")
        
        print(f"Loading PEFT model from {pretrained_model_name_or_path}")
        print(f"Base model: {base_model_name}")
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            **model_kwargs
        )
        
        peft_model = PeftModel.from_pretrained(base_model, pretrained_model_name_or_path)

        has_lora_parameters = any("lora_" in name for name, _ in peft_model.named_parameters())
        if not has_lora_parameters:
            raise RuntimeError(
                "Loaded FastVLM checkpoint without LoRA adapters. "
                "Ensure the adapter_config target modules match the training setup."
            )

        model = peft_model.merge_and_unload()

        custom_proj_path = os.path.join(pretrained_model_name_or_path, "custom_text_proj.pt")
        if os.path.exists(custom_proj_path):
            custom_text_proj_state = torch.load(custom_proj_path, map_location="cpu")
            print(f"Loaded custom_text_proj from {custom_proj_path}")
        else:
            adapter_weights_path = os.path.join(pretrained_model_name_or_path, "adapter_model.safetensors")
            if os.path.exists(adapter_weights_path):
                with safe_open(adapter_weights_path, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        if "custom_text_proj.weight" in key:
                            custom_text_proj_state["weight"] = f.get_tensor(key)
                            print("Found custom_text_proj.weight in adapter")
                        elif "custom_text_proj.bias" in key:
                            custom_text_proj_state["bias"] = f.get_tensor(key)
                            print("Found custom_text_proj.bias in adapter")
            if not custom_text_proj_state:
                warnings.warn(
                    "custom_text_proj.pt not found alongside the LoRA adapters. "
                    "Falling back to randomly initialized projection head.",
                    RuntimeWarning,
                )
    else:
        # Standard model loading
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path,
            **model_kwargs
        )
        
        # Check if there's a separate custom_text_proj file
        custom_proj_path = os.path.join(pretrained_model_name_or_path, "custom_text_proj.pt")
        if os.path.exists(custom_proj_path):
            custom_text_proj_state = torch.load(custom_proj_path, map_location="cpu")
            print(f"Loaded custom_text_proj from {custom_proj_path}")
    
    return model, custom_text_proj_state


class ColFastVLM(nn.Module):
    """Return ColBERT-style embeddings on top of the Hugging Face FastVLM checkpoints."""

    main_input_name: ClassVar[str] = "doc_input_ids"

    def __init__(
        self,
        pretrained_model_name_or_path: str = "apple/FastVLM-0.5B",
        mask_non_image_embeddings: bool = False,
        dim: int = 128,
        fuse_in_decoder: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        
        # Load model with PEFT support
        self.model, custom_text_proj_state = _load_model_with_peft_support(
            pretrained_model_name_or_path,
            **kwargs
        )
        
        self.config = self.model.config
        self.mask_non_image_embeddings = mask_non_image_embeddings
        self.dim = dim
        self.fuse_in_decoder = fuse_in_decoder
        
        # Initialize custom_text_proj layer
        self.custom_text_proj = nn.Linear(self.config.hidden_size, dim)
        
        # Load saved weights if available, otherwise initialize
        if custom_text_proj_state:
            if "weight" in custom_text_proj_state:
                self.custom_text_proj.weight.data = custom_text_proj_state["weight"]
            if "bias" in custom_text_proj_state:
                self.custom_text_proj.bias.data = custom_text_proj_state["bias"]
            print("Loaded custom_text_proj weights from checkpoint")
        else:
            nn.init.normal_(self.custom_text_proj.weight, std=0.02)
            nn.init.zeros_(self.custom_text_proj.bias)
            print("Initialized custom_text_proj with random weights")
        
        # Move to correct device and dtype
        param = next(self.model.parameters())
        self.custom_text_proj = self.custom_text_proj.to(device=param.device, dtype=param.dtype)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> torch.Tensor:
        kwargs.pop("output_hidden_states", None)
        kwargs.pop("return_dict", None)
        text_only_input_ids = kwargs.pop("text_input_ids", None)

        if images is not None:
            images = images.to(device=self.device, dtype=self.model.dtype)

        use_fused = images is not None and self.fuse_in_decoder
        visual_embeddings = None
        visual_mask = None
        hidden = None
        attn_mask = attention_mask

        if use_fused:
            cached_features = {"value": None}
            original_encode = self.model.encode_images

            def _encode_and_cache(model_self, imgs):
                feats = original_encode(imgs)
                cached_features["value"] = feats
                return feats

            self.model.encode_images = MethodType(_encode_and_cache, self.model)
            try:
                _, fused_position_ids, fused_attention_mask, fused_past_key_values, fused_inputs_embeds, _ = (
                    self.model.prepare_inputs_labels_for_multimodal(
                        input_ids=input_ids,
                        position_ids=position_ids,
                        attention_mask=attention_mask,
                        past_key_values=past_key_values,
                        labels=labels,
                        images=images,
                        image_sizes=image_sizes or self._infer_image_sizes(images),
                    )
                )

                outputs = self.model.model(
                    input_ids=None,
                    attention_mask=fused_attention_mask,
                    position_ids=fused_position_ids,
                    past_key_values=fused_past_key_values,
                    inputs_embeds=fused_inputs_embeds,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=True,
                    return_dict=True,
                )
                hidden = outputs.hidden_states[-1]
                attn_mask = fused_attention_mask
                visual_embeddings = cached_features["value"]

                if self.mask_non_image_embeddings and visual_embeddings is not None:
                    visual_mask = self._build_visual_mask(
                        input_ids=input_ids,
                        input_attention_mask=attention_mask,
                        new_attention_mask=fused_attention_mask,
                        visual_features=visual_embeddings,
                    )
            except Exception as error:  # pragma: no cover - defensive fallback
                warnings.warn(
                    "Fused decoder path failed; falling back to non-fused FastVLM processing. "
                    "Set fuse_in_decoder=False to silence this warning.\n"
                    f"Error: {error}"
                )
                use_fused = False
                hidden = None
                visual_embeddings = None
                attn_mask = attention_mask
            finally:
                self.model.encode_images = original_encode

        if hidden is None:
            if images is not None:
                try:
                    visual_embeddings = self.model.encode_images(images)
                    if visual_embeddings.dim() == 2:
                        bsz = images.size(0)
                        npatch = visual_embeddings.size(0) // bsz
                        visual_embeddings = visual_embeddings.reshape(bsz, npatch, -1)
                except (AttributeError, RuntimeError) as error:  # pragma: no cover - defensive fallback
                    warnings.warn(
                        f"Failed to process visual embeddings: {error}\n"
                        f"Vision tower available: {hasattr(self.model.model, 'vision_tower')}\n"
                        f"MM projector available: {hasattr(self.model.model, 'mm_projector')}\n"
                        "Continuing in text-only mode."
                    )
                    visual_embeddings = None
            text_input_ids = text_only_input_ids if text_only_input_ids is not None else input_ids
            text_out = self.model(
                input_ids=text_input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=True,
                images=None,
                image_sizes=None,
                return_dict=True,
                **kwargs,
            )
            text_hidden = text_out.hidden_states[-1]
            if visual_embeddings is not None:
                hidden = torch.cat([text_hidden, visual_embeddings], dim=1)
                if attention_mask is not None:
                    vmask = torch.ones(
                        visual_embeddings.size(0),
                        visual_embeddings.size(1),
                        device=attention_mask.device,
                        dtype=attention_mask.dtype,
                    )
                    attn_mask = torch.cat([attention_mask, vmask], dim=1)
                else:
                    attn_mask = None
                if self.mask_non_image_embeddings:
                    visual_mask = torch.zeros(
                        visual_embeddings.size(0),
                        hidden.size(1),
                        device=hidden.device,
                        dtype=attn_mask.dtype if attn_mask is not None else hidden.dtype,
                    )
                    visual_mask[:, -visual_embeddings.size(1) :] = 1
            else:
                hidden = text_hidden
                attn_mask = attention_mask

        proj = self.custom_text_proj(hidden)
        proj = proj / proj.norm(dim=-1, keepdim=True)
        if attn_mask is not None:
            proj = proj * attn_mask.to(proj.dtype).unsqueeze(-1)

        if self.mask_non_image_embeddings and visual_mask is not None:
            proj = proj * visual_mask.to(proj.dtype).unsqueeze(-1)

        return proj

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        return cls(pretrained_model_name_or_path=pretrained_model_name_or_path, **kwargs)

    def save_pretrained(self, save_directory: str, **kwargs):
        self.model.save_pretrained(save_directory, **kwargs)
        import os

        torch.save(self.custom_text_proj.state_dict(), os.path.join(save_directory, "custom_text_proj.pt"))

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.model.set_output_embeddings(new_embeddings)

    def resize_token_embeddings(
        self,
        new_num_tokens: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
    ) -> nn.Embedding:
        model_embeds = self.model.resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        self.config.vocab_size = model_embeds.num_embeddings
        return model_embeds

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        if hasattr(self.model, "gradient_checkpointing_disable"):
            self.model.gradient_checkpointing_disable()

    @property
    def patch_size(self) -> int:
        if hasattr(self.model.model, "vision_tower") and hasattr(self.model.model.vision_tower, "config"):
            vision_config = self.model.model.vision_tower.config
            if isinstance(vision_config, dict) and "image_cfg" in vision_config:
                return vision_config["image_cfg"].get("patch_size", 64)
        return 64

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def _infer_image_sizes(self, images: Optional[torch.Tensor]) -> Optional[list[list[int]]]:
        if images is None:
            return None
        if isinstance(images, torch.Tensor) and images.ndim == 4:
            height, width = images.shape[-2:]
            return [[int(width), int(height)] for _ in range(images.size(0))]
        return None

    def _build_visual_mask(
        self,
        *,
        input_ids: Optional[torch.LongTensor],
        input_attention_mask: Optional[torch.Tensor],
        new_attention_mask: Optional[torch.Tensor],
        visual_features: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if (
            input_ids is None
            or new_attention_mask is None
            or visual_features is None
            or not isinstance(visual_features, torch.Tensor)
        ):
            return None

        if visual_features.dim() == 2:
            visual_features = visual_features.unsqueeze(0)
        if visual_features.dim() != 3:
            return None

        padding_side = getattr(self.model.config, "tokenizer_padding_side", "right")
        sentinel = getattr(self.model.config, "image_token_index", IMAGE_TOKEN_INDEX)

        batch_size, max_len = new_attention_mask.shape
        mask = torch.zeros(batch_size, max_len, dtype=new_attention_mask.dtype, device=new_attention_mask.device)

        for batch_idx in range(batch_size):
            if input_attention_mask is None:
                active_ids = input_ids[batch_idx]
            else:
                active_ids = input_ids[batch_idx][input_attention_mask[batch_idx].bool()]

            image_positions = torch.where(active_ids == sentinel)[0].tolist()
            if not image_positions:
                continue

            position = image_positions[0]
            seq_len = int(new_attention_mask[batch_idx].sum().item())
            start = max_len - seq_len if padding_side == "left" else 0
            cursor = start + max(position, 0)

            features = visual_features[min(batch_idx, visual_features.size(0) - 1)]
            vision_len = features.shape[0]
            end = min(cursor + vision_len, max_len)
            mask[batch_idx, cursor:end] = 1

        return mask if mask.any() else None
