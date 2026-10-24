"""Exercise request contracts with local stubs, never a paid model call."""
import base64
import sys
from types import SimpleNamespace

import numpy as np
import pytest

import live
from lab import CORPUS


@pytest.fixture
def configured(monkeypatch):
    monkeypatch.setenv("MM_RUN_LIVE", "1")
    monkeypatch.setenv("VLM_MODEL", "openai/test-vision-alias")
    monkeypatch.setenv("VLM_API_KEY", "test-value-not-a-real-key")
    monkeypatch.setenv("EMBED_MODEL", "openai/test-embedding-alias")
    monkeypatch.setenv("EMBED_API_KEY", "test-value-not-a-real-key")


def test_optional_models_require_explicit_switch(monkeypatch):
    monkeypatch.setenv("MM_RUN_MODELS", "0")
    with pytest.raises(RuntimeError, match="MM_RUN_MODELS"):
        live.FastEmbedModels()


def test_remote_calls_require_explicit_switch(monkeypatch):
    monkeypatch.setenv("MM_RUN_LIVE", "0")
    with pytest.raises(RuntimeError, match="MM_RUN_LIVE"):
        live.api_options()


def test_missing_embedding_model_does_not_reuse_vision_model(configured, monkeypatch):
    monkeypatch.delenv("EMBED_MODEL")
    with pytest.raises(RuntimeError, match="EMBED_MODEL"):
        live.api_options("EMBED")


def test_image_attachment_round_trip_uses_actual_pixels(tmp_path):
    path = tmp_path / "sample.png"
    path.write_bytes(b"controlled test payload")
    part = live.image_part(path)
    prefix, encoded = part["image_url"]["url"].split(",")
    assert prefix == "data:image/png;base64"
    assert base64.b64decode(encoded) == path.read_bytes()
    with pytest.raises(ValueError):
        live.image_part(tmp_path / "report.pdf")


def test_reader_sends_selected_pixels_and_frame_time_only(configured, monkeypatch, tmp_path):
    path = tmp_path / "frame.png"
    path.write_bytes(b"selected frame")
    captured = {}
    def completion(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="stubbed response"))])
    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(completion=completion))
    bundle = [{"source_id": "video#segment1", "text": "observed source", "meta": {"start": 4, "end": 8}, "images": [{"path": str(path), "sample_time": 6}]}]
    assert live.answer_from_bundle("What is visible?", bundle) == "stubbed response"
    assert captured["model"] == "openai/test-vision-alias"
    parts = captured["messages"][1]["content"]
    assert len([p for p in parts if p["type"] == "image_url"]) == 1
    assert "6.0 seconds" in str(parts)
    assert "answer_key" not in str(captured)


def test_empty_evidence_does_not_call_model(monkeypatch):
    monkeypatch.setenv("MM_RUN_LIVE", "0")
    assert "cannot answer" in live.answer_from_bundle("q", [])


def test_embedding_response_order_is_restored(configured, monkeypatch):
    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(embedding=lambda **kwargs: SimpleNamespace(data=[{"index": 1, "embedding": [0, 2]}, {"index": 0, "embedding": [3, 0]}])))
    np.testing.assert_allclose(live.embed_via_api(["a", "b"]), [[1, 0], [0, 1]])


@pytest.mark.parametrize("rows", [
    [{"index":0,"embedding":[1,0]},{"index":0,"embedding":[0,1]}],
    [{"index":0,"embedding":[1,0]},{"index":2,"embedding":[0,1]}],
    [{"index":0,"embedding":[0,0]},{"index":1,"embedding":[0,1]}],
    [{"index":0,"embedding":[float("nan"),0]},{"index":1,"embedding":[0,1]}],
])
def test_malformed_hosted_vectors_are_rejected(configured, monkeypatch, rows):
    monkeypatch.setitem(sys.modules,"litellm",SimpleNamespace(embedding=lambda **kwargs:SimpleNamespace(data=rows)))
    with pytest.raises(ValueError):
        live.embed_via_api(["a","b"])
