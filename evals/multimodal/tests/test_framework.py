"""Verify LlamaIndex's actual adapter contract with controlled local vectors.

This checks integration plumbing, not pretrained model quality.
"""
import importlib.util
import sys
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.mark.skipif(importlib.util.find_spec("fastembed") is None,
                    reason="Install --extra models for FastEmbed registry checks")
def test_installed_fastembed_registers_the_requested_paired_towers():
    from fastembed import TextEmbedding, ImageEmbedding
    text = {row["model"]:row for row in TextEmbedding.list_supported_models()}
    image = {row["model"]:row for row in ImageEmbedding.list_supported_models()}
    assert text["BAAI/bge-small-en-v1.5"]["dim"] == 384
    assert text["Qdrant/clip-ViT-B-32-text"]["dim"] == 512
    assert image["Qdrant/clip-ViT-B-32-vision"]["dim"] == 512
    assert callable(TextEmbedding.query_embed)
    assert callable(TextEmbedding.passage_embed)


@pytest.mark.skipif(importlib.util.find_spec("llama_index") is None,
                    reason="Install --extra models for real framework contract checks")
def test_actual_llamaindex_accepts_both_adapters_without_network(monkeypatch):
    monkeypatch.setenv("MM_RUN_MODELS", "1")

    class ControlledEncoder:
        def __init__(self, model_name, **kwargs):
            self.dimension = 512 if "clip" in model_name else 384

        def embed(self, values):
            for value in values:
                row = np.zeros(self.dimension)
                row[0] = 1
                yield row

        passage_embed = embed
        query_embed = embed

    monkeypatch.setitem(sys.modules, "fastembed", SimpleNamespace(
        TextEmbedding=ControlledEncoder, ImageEmbedding=ControlledEncoder))
    from framework import build_framework_index
    from llama_index.core.schema import ImageNode

    index = build_framework_index()
    hits = index.as_retriever(similarity_top_k=2, image_similarity_top_k=2).retrieve("solar")
    assert len(hits) == 4
    assert sum(isinstance(hit.node, ImageNode) for hit in hits) == 2
    assert all(hit.node.metadata["source_id"] for hit in hits)


def test_all_strategy_paths_execute_with_controlled_vectors(monkeypatch):
    import live

    class ControlledModels:
        def _rows(self, values):
            return np.tile([1.,0.,0.], (len(values),1))
        text_documents = _rows
        text_queries = _rows
        clip_texts = _rows
        clip_images = _rows

    monkeypatch.setattr(live,"FastEmbedModels",ControlledModels)
    results = live.run_model_comparison(k=2)
    assert len(results) == 3
    for summary, rows in results.values():
        assert summary["supported_queries"] == 6
        assert summary["unsupported_queries"] == 2
        assert len(rows) == 8
        assert all(len(row["retrieved"]) <= 2 for row in rows)
