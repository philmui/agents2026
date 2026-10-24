"""Behavior checks for the lesson's mathematical and evidence boundaries."""
from dataclasses import asdict
from pathlib import Path
import copy

import numpy as np
import pytest

from lab import (
    CORPUS, ColorEncoder, LexicalEncoder, QdrantIndex, Record,
    build_strategy_a, build_video_records, check_data_points, evaluate,
    evidence_bundle, feature_fingerprint, load_corpus, load_reference_parses,
    normalize, read_pdf, rrf, sample_keyframes, timestamp, validate_parse,
)


def hit(identifier, source=None, modality="text", **extra):
    return {"id": identifier, "source_id": source or identifier,
            "modality": modality, "path": "unused", "text": identifier,
            "score": 0.9, "meta": {}, **extra}


def test_unit_vectors_preserve_direction_and_zero_is_explicit():
    rows = normalize([[3, 4], [30, 40], [0, 0]])
    np.testing.assert_allclose(rows, [[0.6, 0.8], [0.6, 0.8], [0, 0]])
    assert np.dot(rows[0], rows[1]) == pytest.approx(1)
    assert np.isfinite(rows).all()


def test_unseen_words_do_not_become_fake_semantic_matches():
    encoder = LexicalEncoder(["solar panel", "water cleaning"])
    np.testing.assert_array_equal(encoder.embed(["lunar harvest"]), np.zeros((1, encoder.dimension)))


def test_qdrant_retrieval_filter_and_original_source_survive():
    records = [Record("memo", "text", "memo.md"), Record("picture", "image", "picture.png")]
    index = QdrantIndex(records, [[1, 0], [0.8, 0.6]])
    assert index.search([1, 0], 1)[0]["id"] == "memo"
    found = index.search([1, 0], 1, modality="image")
    assert found[0]["id"] == "picture"
    assert found[0]["path"] == "picture.png"
    assert index.search([0, 0]) == []
    index.client.close()


@pytest.mark.parametrize("vectors", [[[0, 0]], [[np.nan, 1]], [[np.inf, 1]], [[1, 0], [0, 1]]])
def test_bad_stored_vectors_fail_before_indexing(vectors):
    with pytest.raises(ValueError):
        QdrantIndex([Record("a", "text", "a")], vectors)


def test_wrong_query_dimension_is_rejected():
    index = QdrantIndex([Record("a", "text", "a")], [[1, 0]])
    with pytest.raises(ValueError):
        index.search([1, 2, 3])
    index.client.close()


def test_rrf_uses_ranks_not_incompatible_scores():
    lists = [[hit("a", score=1e6), hit("b", score=100)], [hit("b", score=.01), hit("c", score=.009)]]
    found = rrf(lists)
    assert found[0]["source_id"] == "b"
    assert found[0]["score"] == pytest.approx(1/62+1/61)


def test_duplicate_frames_do_not_buy_extra_votes():
    found = rrf([[hit("frame1", "seg0"), hit("frame2", "seg0"), hit("frame3", "seg1")]])
    assert found[0]["score"] == pytest.approx(1/61)
    assert found[1]["score"] == pytest.approx(1/62)
    assert len(found[0]["evidence"]) == 2
    assert len(found[0]["votes"]) == 1


def test_transcript_and_frame_can_vote_for_same_segment():
    found = rrf([[hit("transcript0", "seg0", "video_text")], [hit("frame0", "seg0", "video_frame")]])
    assert len(found) == 1
    assert found[0]["score"] == pytest.approx(2/61)
    assert {e["modality"] for e in found[0]["evidence"]} == {"video_text", "video_frame"}


def test_rrf_empty_lists_ties_and_negative_constant():
    assert rrf([[], []]) == []
    assert [h["id"] for h in rrf([[hit("z")], [hit("a")]])] == ["a", "z"]
    with pytest.raises(ValueError):
        rrf([], constant=-1)


def test_recall_hit_rate_and_rr_are_distinct():
    summary, rows = evaluate(lambda query, k: [hit("wrong"), hit("a")],
                             [{"query": "q", "relevant": {"a", "b"}}], k=2)
    assert summary["Recall@2"] == .5
    assert summary["HitRate@2"] == 1
    assert summary["MRR@2"] == .5
    assert rows[0]["relevant"] == ["a", "b"]


def test_unsupported_query_has_undefined_recall_separate_return_rate():
    summary, rows = evaluate(lambda q, k: [hit("a")], [{"query": "future", "relevant": set()}])
    assert summary["Recall@3"] is None
    assert summary["unsupported_evidence_return_rate"] == 1
    assert rows[0]["rr"] is None


def test_parse_shape_does_not_prove_correct_label_value_pairing():
    parsed = copy.deepcopy(load_reference_parses()["chart_energy"])
    validate_parse(parsed)
    expected = copy.deepcopy(parsed["data_points"])
    parsed["data_points"][0]["value"], parsed["data_points"][1]["value"] = (
        parsed["data_points"][1]["value"], parsed["data_points"][0]["value"])
    assert sorted(p["value"] for p in parsed["data_points"]) == sorted(p["value"] for p in expected)
    validate_parse(parsed)  # A schema checker cannot see the pixels.
    check = check_data_points(parsed["data_points"], expected)
    assert not check["ok"] and len(check["mismatched"]) == 2


def test_units_and_duplicate_labels_are_checked():
    expected = [{"label": "May", "value": 12, "unit": "kWh"}]
    assert not check_data_points([{**expected[0], "unit": "Wh"}], expected)["ok"]
    assert not check_data_points(expected*2, expected)["ok"]


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), "12"])
def test_parse_rejects_non_numeric_or_nonfinite_values(value):
    parsed = {"kind": "chart", "title": "x", "summary": "x", "data_points": [{"label": "May", "unit": "kWh", "value": value}]}
    with pytest.raises(ValueError):
        validate_parse(parsed)


def test_color_model_reads_pixels_but_cannot_read_numbers():
    images = [r for r in load_corpus() if r.modality == "image"]
    encoder = ColorEncoder()
    vectors = encoder.embed_images([r.path for r in images])
    assert vectors.shape == (4, 4)
    assert np.isfinite(vectors).all()
    np.testing.assert_array_equal(encoder.embed_text(["highest solar output"]), [[0, 0, 0, 0]])


def test_raw_images_have_no_answer_text_and_descriptions_are_labeled():
    raw = load_corpus()
    assert all(not r.text for r in raw if r.modality == "image")
    records, encoder, index = build_strategy_a()
    assert all(r.meta.get("parse_source") for r in records if r.modality == "image")
    assert index.search(encoder.embed(["controller E17 warning"])[0], 3)
    index.client.close()


def test_feature_identity_changes_on_source_model_and_preprocessing(tmp_path):
    source = tmp_path / "image.txt"
    source.write_text("pixels-v1")
    record = Record("source", "image", str(source))
    def fingerprint(model="clip-v1", prep="resize224"):
        return feature_fingerprint(record, model, "parse-v1", prep)["feature_key"]
    initial = fingerprint()
    assert initial != fingerprint("clip-v2")
    assert initial != fingerprint(prep="crop224")
    source.write_text("pixels-v2")
    assert initial != fingerprint()


def test_half_open_windows_do_not_borrow_unrelated_frames():
    segments = [{"id": "s0", "start": 0, "end": 4, "text": "first"},
                {"id": "s1", "start": 4, "end": 8, "text": "second"}]
    frames = [{"id": "f4", "time": 4, "path": "frame4.png"},
              {"id": "f9", "time": 9, "path": "frame9.png"}]
    texts, visuals = build_video_records(segments, frames)
    assert len(visuals) == 1
    assert visuals[0].source_id == texts[1].source_id
    first_bundle = evidence_bundle([{**asdict(texts[0]), "score": 1}], texts, visuals)
    assert first_bundle[0]["images"] == []


def test_evidence_bundle_contains_only_selected_segment_and_budget():
    texts = [Record("t0", "video_text", "video", "first", "s0"), Record("t1", "video_text", "video", "secret second", "s1")]
    frames = [Record("f0", "video_frame", "0.png", source_id="s0", meta={"sample_time": 1}), Record("f1", "video_frame", "1.png", source_id="s1", meta={"sample_time": 5})]
    bundle = evidence_bundle([hit("t0", "s0", "video_text")], texts, frames, max_images=1)
    assert bundle[0]["text"] == "first"
    assert len(bundle[0]["images"]) == 1
    assert bundle[0]["images"][0]["path"] == "0.png"
    assert bundle[0]["images"][0]["sample_time"] == 1
    assert "secret second" not in str(bundle)
    assert evidence_bundle([hit("t0", "s0", "video_text")], texts, frames, max_images=0)[0]["images"] == []


def test_pdf_native_text_and_image_only_page_have_distinct_evidence(tmp_path):
    pages = read_pdf(CORPUS / "solar_lab_report.pdf", tmp_path / "pages")
    assert len(pages) == 2
    assert pages[0].text.strip()
    assert not pages[1].text.strip()
    assert all(Path(p.path).exists() and p.meta["page"] == i for i, p in enumerate(pages, 1))


def test_video_is_decoded_with_timestamped_unique_frames(tmp_path):
    frames = sample_keyframes(CORPUS / "solar_lab_review.avi", 2, tmp_path / "frames")
    assert len(frames) == 6
    assert [f["time"] for f in frames] == [0, 2, 4, 6, 8, 10]
    assert len({f["id"] for f in frames}) == 6
    with pytest.raises(ValueError):
        sample_keyframes(CORPUS / "missing.avi", output_dir=tmp_path / "missing")
    with pytest.raises(ValueError):
        sample_keyframes(CORPUS / "solar_lab_review.avi", 0)


def test_reader_preserves_the_retrieved_frame_instead_of_earlier_scene():
    frames = [Record("before", "video_frame", "before.png", source_id="s0", meta={"sample_time":0}),
              Record("warning", "video_frame", "warning.png", source_id="s0", meta={"sample_time":2})]
    selected = {**asdict(frames[1]), "score":.9}
    bundle = evidence_bundle([selected], frame_records=frames)
    assert bundle[0]["images"][0]["path"] == "warning.png"
    assert bundle[0]["images"][0]["sample_time"] == 2
    # Fusion may retain the transcript as its representative hit, so inspect
    # its visual evidence list too; do not lose the actual retrieved pixels.
    fused = rrf([[hit("text", "s0", "video_text")], [selected]])
    bundle = evidence_bundle(fused, frame_records=frames)
    assert bundle[0]["images"][0]["path"] == "warning.png"


def test_pdf_render_identity_prevents_overwriting_another_source(tmp_path):
    from reportlab.pdfgen.canvas import Canvas
    paths = []
    for folder, text in [("first", "First source"), ("second", "Second source")]:
        directory = tmp_path / folder
        directory.mkdir()
        path = directory / "report.pdf"
        pdf = Canvas(str(path))
        pdf.drawString(50, 700, text)
        pdf.save()
        paths.append(path)
    pages1 = read_pdf(paths[0], tmp_path / "renders")
    saved = Path(pages1[0].path).read_bytes()
    pages2 = read_pdf(paths[1], tmp_path / "renders")
    assert pages1[0].path != pages2[0].path
    assert pages1[0].id != pages2[0].id
    assert Path(pages1[0].path).read_bytes() == saved


def test_timestamp_rounding_carries_to_next_minute():
    assert timestamp(59.96) == "01:00.0"
    assert timestamp(119.96) == "02:00.0"
    assert timestamp(4.0) == "00:04.0"
