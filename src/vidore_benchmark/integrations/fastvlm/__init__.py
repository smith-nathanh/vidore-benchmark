"""FastVLM integration used by Vidore benchmark retrievers."""

from .modeling_fastvlm import ColFastVLM
from .processing_fastvlm import ColFastVLMProcessor

__all__ = ["ColFastVLM", "ColFastVLMProcessor"]
