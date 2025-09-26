"""Tests for FastVLM retriever including PEFT checkpoint loading."""

import os
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest
import torch
from PIL import Image

from vidore_benchmark.retrievers.fastvlm_retriever import FastVLMRetriever
from vidore_benchmark.utils.torch_utils import tear_down_torch


@pytest.fixture(scope="module")
def retriever() -> Generator[FastVLMRetriever, None, None]:
    """Base FastVLM retriever fixture."""
    yield FastVLMRetriever()
    tear_down_torch()


@pytest.fixture(scope="module")
def peft_checkpoint_path() -> str:
    """Path to PEFT checkpoint if it exists locally."""
    local_path = os.path.expanduser("~/models/fastvlm/checkpoint-3694")
    if os.path.exists(local_path):
        return local_path
    return None


@pytest.mark.slow
def test_forward_queries(retriever: FastVLMRetriever, queries_fixture):
    """Test query encoding produces correct output shape."""
    embedding_queries = retriever.forward_queries(queries_fixture, batch_size=1)
    assert len(embedding_queries) == len(queries_fixture)
    assert embedding_queries[0].shape[1] == 128
    # Check embeddings are normalized
    for emb in embedding_queries:
        norms = torch.norm(emb, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


@pytest.mark.slow
def test_forward_documents(retriever: FastVLMRetriever, image_passage_fixture):
    """Test document encoding produces correct output shape."""
    embedding_docs = retriever.forward_passages(image_passage_fixture, batch_size=1)
    assert len(embedding_docs) == len(image_passage_fixture)
    assert embedding_docs[0].shape[1] == 128
    # Check embeddings are normalized
    for emb in embedding_docs:
        norms = torch.norm(emb, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


@pytest.mark.slow
def test_get_scores(
    retriever: FastVLMRetriever,
    query_multi_vector_embeddings_fixture,
    passage_multi_vector_embeddings_fixture,
):
    """Test scoring function produces correct output shape."""
    scores = retriever.get_scores(
        query_multi_vector_embeddings_fixture,
        passage_multi_vector_embeddings_fixture,
    )
    assert scores.shape == (
        len(query_multi_vector_embeddings_fixture),
        len(passage_multi_vector_embeddings_fixture),
    )


@pytest.mark.slow
def test_mixed_batch_processing(retriever: FastVLMRetriever):
    """Test processing different sized images in a batch."""
    # Create images of different sizes
    images = [
        Image.new("RGB", (224, 224), color="red"),
        Image.new("RGB", (512, 384), color="green"),
        Image.new("RGB", (1024, 768), color="blue"),
    ]

    embedding_docs = retriever.forward_passages(images, batch_size=2)
    assert len(embedding_docs) == len(images)
    # All should have same embedding dimension
    for emb in embedding_docs:
        assert emb.shape[1] == 128


@pytest.mark.slow
def test_empty_input_handling(retriever: FastVLMRetriever):
    """Test retriever handles empty inputs gracefully."""
    # Test empty queries
    with pytest.raises((ValueError, IndexError)):
        retriever.forward_queries([], batch_size=1)

    # Test empty passages
    with pytest.raises((ValueError, IndexError)):
        retriever.forward_passages([], batch_size=1)


@pytest.mark.slow
def test_single_query_document_pair(retriever: FastVLMRetriever):
    """Test retriever with single query and document."""
    query = ["What is shown in this image?"]
    image = [Image.new("RGB", (256, 256), color="white")]

    query_emb = retriever.forward_queries(query, batch_size=1)
    doc_emb = retriever.forward_passages(image, batch_size=1)

    scores = retriever.get_scores(query_emb, doc_emb)
    assert scores.shape == (1, 1)
    assert isinstance(scores[0, 0].item(), float)


@pytest.mark.slow
@pytest.mark.skipif(
    not os.path.exists(os.path.expanduser("~/models/fastvlm/checkpoint-3694")),
    reason="PEFT checkpoint not available locally"
)
def test_peft_checkpoint_loading(peft_checkpoint_path):
    """Test loading FastVLM with PEFT/LoRA checkpoint."""
    if not peft_checkpoint_path:
        pytest.skip("PEFT checkpoint not found")

    # Load retriever with PEFT checkpoint
    retriever = FastVLMRetriever(
        pretrained_model_name_or_path=peft_checkpoint_path,
        device="cpu",
        torch_dtype=torch.float32,
    )

    # Test basic functionality
    query = ["test query"]
    query_emb = retriever.forward_queries(query, batch_size=16)

    assert query_emb[0].shape[1] == 128
    assert not torch.isnan(query_emb[0]).any()
    assert not torch.isinf(query_emb[0]).any()

    tear_down_torch()


@pytest.mark.slow
def test_processor_configuration(retriever: FastVLMRetriever):
    """Test processor is configured correctly."""
    processor = retriever.processor

    # Check image processor settings for FastVLM/MobileCLIP
    assert processor.image_processor.size["shortest_edge"] == 1024
    assert processor.image_processor.crop_size["height"] == 1024
    assert processor.image_processor.crop_size["width"] == 1024

    # Check normalization settings
    assert processor.image_processor.image_mean == [0.0, 0.0, 0.0]
    assert processor.image_processor.image_std == [1.0, 1.0, 1.0]

    # Check tokenizer padding
    assert processor.tokenizer.padding_side == "left"


@pytest.mark.slow
def test_model_dtype_consistency(retriever: FastVLMRetriever):
    """Test model components have consistent dtypes."""
    model = retriever.model

    # Check main model dtype
    model_dtype = next(model.model.parameters()).dtype

    # Check custom projection layer dtype
    proj_dtype = model.custom_text_proj.weight.dtype

    assert model_dtype == proj_dtype, f"Dtype mismatch: model={model_dtype}, proj={proj_dtype}"


@pytest.mark.slow
def test_batch_size_variations(retriever: FastVLMRetriever, image_passage_fixture):
    """Test retriever with different batch sizes."""
    # Create more images for testing
    images = image_passage_fixture * 3  # 6 images total

    # Test different batch sizes
    for batch_size in [1, 2, 3, 6]:
        embeddings = retriever.forward_passages(images, batch_size=batch_size)
        assert len(embeddings) == len(images)

        # All embeddings should be identical for same input regardless of batch size
        for emb in embeddings:
            assert emb.shape[1] == 128


# Unit tests (faster, don't load full model)

def test_retriever_initialization():
    """Test retriever can be initialized with different configs."""
    with patch("vidore_benchmark.retrievers.fastvlm_retriever.ColFastVLM") as mock_model:
        with patch("vidore_benchmark.retrievers.fastvlm_retriever.ColFastVLMProcessor") as mock_processor:
            # Mock the model and processor
            mock_instance = MagicMock()
            mock_model.from_pretrained.return_value = mock_instance
            mock_proc_instance = MagicMock()
            mock_processor.from_pretrained.return_value = mock_proc_instance

            # Test default initialization
            retriever = FastVLMRetriever()
            mock_model.from_pretrained.assert_called_once()
            assert "apple/FastVLM-0.5B" in str(mock_model.from_pretrained.call_args)

            # Test custom model initialization
            retriever = FastVLMRetriever(
                pretrained_model_name_or_path="custom/model",
                device="cuda",
            )
            assert "custom/model" in str(mock_model.from_pretrained.call_args)


def test_retriever_properties():
    """Test retriever property access."""
    with patch("vidore_benchmark.retrievers.fastvlm_retriever.ColFastVLM") as mock_model:
        with patch("vidore_benchmark.retrievers.fastvlm_retriever.ColFastVLMProcessor") as mock_processor:
            mock_instance = MagicMock()
            mock_instance.eval.return_value = mock_instance
            mock_model.from_pretrained.return_value = mock_instance

            mock_proc_instance = MagicMock()
            mock_processor.from_pretrained.return_value = mock_proc_instance

            retriever = FastVLMRetriever()

            # Test that model and processor are accessible
            assert retriever.model is not None
            assert retriever.processor is not None
            assert retriever.device is not None
            assert retriever.use_visual_embedding is True