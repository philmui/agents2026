"""Transparent, offline building blocks for the multimodal retrieval lessons.

Default retrieval uses an explicit bag-of-words encoder and a deliberately tiny
color encoder. Neither is a pretrained neural model. Qdrant really stores and
searches their vectors. No function here calls a remote service.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
import hashlib
import json
import re
import uuid

import numpy as np
from PIL import Image
from qdrant_client import QdrantClient, models

ROOT = Path(__file__).resolve().parent
CORPUS = ROOT / "assets" / "corpus"


@dataclass
class Record:
    id: str
    modality: str
    path: str
    text: str = ""
    source_id: str = ""
    parent_id: str = ""
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.source_id = self.source_id or self.id
        self.parent_id = self.parent_id or self.source_id


def load_corpus() -> list[Record]:
    """Load raw source files only. Never opens the evaluation answer key."""
    records = []
    for entry in json.loads((CORPUS / "manifest.json").read_text()):
        path = CORPUS / entry["path"]
        records.append(Record(**{**entry, "path": str(path), "text": path.read_text() if entry["modality"] == "text" else ""}))
    return records


def load_reference_parses() -> dict:
    """Hand-authored ingestion fixtures, explicitly not recorded model results."""
    return json.loads((CORPUS / "reference_parses.json").read_text())


def validate_parse(parsed: dict) -> dict:
    """Validate shape, types, and duplicate labels; this cannot prove truth."""
    for key in ("kind", "title", "summary"):
        if not isinstance(parsed.get(key), str):
            raise ValueError(f"{key} must be a string")
    if not isinstance(parsed.get("data_points"), list):
        raise ValueError("data_points must be a list")
    seen = set()
    for point in parsed["data_points"]:
        if not isinstance(point, dict) or not isinstance(point.get("label"), str) or not isinstance(point.get("unit"), str):
            raise ValueError("each data point needs label and unit strings")
        value = point.get("value")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
            raise ValueError("data-point values must be finite numbers")
        label = point["label"].strip().casefold()
        if label in seen:
            raise ValueError(f"duplicate label: {point['label']}")
        seen.add(label)
    return parsed


def flatten_parse(parsed: dict) -> str:
    validate_parse(parsed)
    points = "; ".join(f"{p['label']}: {p['value']} {p['unit']}" for p in parsed["data_points"])
    return f"{parsed['kind']}. {parsed['title']}. {parsed['summary']} {points}".strip()


def described_records(raw=None, parses=None) -> list[Record]:
    raw = load_corpus() if raw is None else raw
    parses = load_reference_parses() if parses is None else parses
    return [Record(**{**asdict(record), "text": flatten_parse(parses[record.id]), "meta": {**record.meta, "parse_source": parses[record.id].get("provenance", "caller supplied parse")}}) if record.modality == "image" else record for record in raw]


def tokens(text: str) -> list[str]:
    """Simple readable tokenizer; no stemming, synonyms, or learned semantics."""
    stop = {"a", "an", "the", "is", "are", "in", "on", "of", "to", "and", "for", "it", "by", "at", "what", "which", "does", "how", "with", "was", "do", "we", "this", "can", "from", "when", "that"}
    return [word for word in re.findall(r"[\w]+", text.casefold()) if word not in stop]


def normalize(vectors):
    array = np.asarray(vectors, dtype=np.float64)
    lengths = np.linalg.norm(array, axis=-1, keepdims=True)
    return np.divide(array, lengths, out=np.zeros_like(array), where=lengths > 0)


class LexicalEncoder:
    """Normalized word counts, with a vocabulary learned from documents only.

    This is a sparse lexical representation stored densely for small lessons.
    Zero vectors remain zero so unsupported vocabulary can be detected.
    """
    def __init__(self, texts):
        self.vocabulary = sorted({word for text in texts for word in tokens(text)})
        if not self.vocabulary:
            raise ValueError("Fit the vocabulary on at least one nonempty document")
        self.lookup = {word: index for index, word in enumerate(self.vocabulary)}
        self.dimension = len(self.vocabulary)

    def embed(self, texts):
        rows = np.zeros((len(texts), self.dimension), dtype=np.float64)
        for row, text in enumerate(texts):
            for word, count in Counter(tokens(text)).items():
                if word in self.lookup:
                    rows[row, self.lookup[word]] = count
        return normalize(rows)


class QdrantIndex:
    """One local in-memory collection with deterministic source metadata.

    UUID5 maps each human ID to a valid Qdrant ID. For this tiny corpus we fetch
    all candidates before sorting ties by human ID; large deployments should
    use an explicit tie-break policy appropriate to their search budget.
    """
    def __init__(self, records, vectors, name="lesson"):
        self.records = list(records)
        matrix = np.asarray(vectors, dtype=float)
        if not self.records or matrix.ndim != 2 or len(matrix) != len(self.records):
            raise ValueError("Need one vector row for every nonempty record list")
        if len({record.id for record in self.records}) != len(self.records):
            raise ValueError("Every indexed chunk needs a unique id")
        if not np.isfinite(matrix).all() or np.any(np.linalg.norm(matrix, axis=1) == 0):
            raise ValueError("Stored vectors must be finite and nonzero")
        self.dimension = matrix.shape[1]
        self.name = name
        self.client = QdrantClient(":memory:")
        self.client.create_collection(name, vectors_config=models.VectorParams(size=self.dimension, distance=models.Distance.COSINE))
        self.client.upsert(name, points=[models.PointStruct(id=str(uuid.uuid5(uuid.NAMESPACE_URL, "cedar-school:"+record.id)), vector=vector.tolist(), payload=asdict(record)) for record, vector in zip(self.records, matrix)])

    def search(self, vector, k=3, modality=None):
        if k < 1:
            return []
        vector = np.asarray(vector, dtype=float)
        if vector.shape != (self.dimension,) or not np.isfinite(vector).all():
            raise ValueError(f"Query must have {self.dimension} finite coordinates")
        if np.linalg.norm(vector) == 0:
            return []
        query_filter = models.Filter(must=[models.FieldCondition(key="modality", match=models.MatchValue(value=modality))]) if modality else None
        points = self.client.query_points(collection_name=self.name, query=vector.tolist(), query_filter=query_filter, limit=len(self.records), with_payload=True).points
        hits = [{**point.payload, "score": float(point.score)} for point in points]
        return sorted(hits, key=lambda hit: (-round(hit["score"], 10), hit["id"]))[:k]


def build_strategy_a(raw=None, parses=None):
    records = described_records(raw, parses)
    encoder = LexicalEncoder([record.text for record in records])
    index = QdrantIndex(records, encoder.embed([record.text for record in records]), "strategy_a")
    return records, encoder, index


class ColorEncoder:
    """A deliberately restricted shared space: [orange, purple, green, teal].

    Pixels are counted near four RGB prototypes; query color words select the
    matching coordinates. This is hand-designed color retrieval, NOT CLIP,
    semantic understanding, OCR, or a neural-model benchmark.
    """
    names = ["orange", "purple", "green", "teal"]
    prototypes = np.array([[216, 106, 40], [134, 100, 197], [58, 148, 110], [38, 143, 153]])

    def embed_images(self, paths):
        rows = []
        for path in paths:
            with Image.open(path) as image:
                pixels = np.asarray(image.convert("RGB").resize((160, 104)), dtype=float).reshape(-1, 3)
            distance = np.linalg.norm(pixels[:, None, :] - self.prototypes[None, :, :], axis=-1)
            nearest = distance.argmin(axis=1)
            selected = distance.min(axis=1) < 65
            rows.append(np.bincount(nearest[selected], minlength=4))
        return normalize(rows)

    def embed_text(self, texts):
        return normalize([[tokens(text).count(name) for name in self.names] for text in texts])


def toy_shared_space(raw=None):
    records = [record for record in (load_corpus() if raw is None else raw) if record.modality == "image"]
    encoder = ColorEncoder()
    index = QdrantIndex(records, encoder.embed_images([record.path for record in records]), "color_only")
    return records, encoder, index


def rrf(ranked_lists, constant=60, source_key="source_id"):
    """Fuse compact unique-source ranks; one vote/source/list, deterministic ties.

    Evidence retains all distinct chunks observed for the source. If several
    frames from a segment appear in one list they count as one source vote.
    Ranking at the segment level does not collapse an entire video into a hit.
    """
    if constant < 0:
        raise ValueError("RRF constant must be nonnegative")
    fused = {}
    for list_index, hits in enumerate(ranked_lists):
        seen = set()
        compact_rank = 0
        for hit in hits:
            source = hit.get(source_key) or hit["id"]
            if source not in fused:
                fused[source] = {**hit, "score": 0.0, "votes": [], "evidence": []}
            row = fused[source]
            if hit["id"] not in {item["id"] for item in row["evidence"]}:
                row["evidence"].append({k: v for k, v in hit.items() if k not in {"evidence", "votes"}})
            if source in seen:
                continue
            seen.add(source)
            compact_rank += 1
            contribution = 1.0 / (constant + compact_rank)
            row["score"] += contribution
            row["votes"].append({"list": list_index, "rank": compact_rank, "contribution": contribution})
    return sorted(fused.values(), key=lambda hit: (-hit["score"], hit.get(source_key) or hit["id"]))


def show_hits(hits):
    """Small plain-Python rows remain readable without pandas."""
    return [{"rank": rank, "id": h["id"], "source": h["source_id"], "modality": h["modality"], "score": round(h["score"], 4)} for rank, h in enumerate(hits, 1)]


GOLD = [
    {"query": "monthly solar energy", "relevant": {"chart_energy", "memo_schedule"}, "kind": "multiple sources"},
    {"query": "purple panel temperature chart", "relevant": {"chart_temperature"}, "kind": "visual and text"},
    {"query": "green water cleaning methods", "relevant": {"chart_water", "memo_cleaning"}, "kind": "multiple sources"},
    {"query": "controller E17 warning", "relevant": {"screenshot_controller", "memo_safety"}, "kind": "exact code and context"},
    {"query": "Thursday classroom review", "relevant": {"memo_schedule"}, "kind": "text"},
    {"query": "orange bar chart", "relevant": {"chart_energy"}, "kind": "visual"},
    {"query": "lunar greenhouse harvest", "relevant": set(), "kind": "unsupported, out of vocabulary"},
    {"query": "solar energy next winter forecast", "relevant": set(), "kind": "unsupported, overlapping vocabulary"},
]


def evaluate(retrieve, questions=GOLD, k=3):
    """Return per-query metrics; no-relevant-source cases are separate.

    Recall@k = unique relevant retrieved / all relevant; Hit@k = any relevant;
    RR@k = reciprocal first relevant rank, or zero if absent within k.
    An unsupported query has undefined recall, so report evidence-return rate
    separately. Returning a hit is not itself a generated false answer.
    """
    rows = []
    for case in questions:
        hits = retrieve(case["query"], k)
        retrieved = list(dict.fromkeys(hit.get("source_id", hit["id"]) for hit in hits))[:k]
        relevant = set(case["relevant"])
        found = relevant & set(retrieved)
        first = next((rank for rank, source in enumerate(retrieved, 1) if source in relevant), None)
        rows.append({"query": case["query"], "retrieved": retrieved, "relevant": sorted(relevant), "recall": len(found)/len(relevant) if relevant else None, "hit": int(bool(found)) if relevant else None, "rr": 1/first if first else (0.0 if relevant else None), "unsupported_return": bool(retrieved) if not relevant else None})
    supported = [row for row in rows if row["recall"] is not None]
    unsupported = [row for row in rows if row["recall"] is None]
    summary = {"k": k, "supported_queries": len(supported), "unsupported_queries": len(unsupported), f"Recall@{k}": float(np.mean([r["recall"] for r in supported])) if supported else None, f"HitRate@{k}": float(np.mean([r["hit"] for r in supported])) if supported else None, f"MRR@{k}": float(np.mean([r["rr"] for r in supported])) if supported else None, "unsupported_evidence_return_rate": float(np.mean([r["unsupported_return"] for r in unsupported])) if unsupported else None}
    return summary, rows


def check_data_points(predicted, expected):
    """Evaluation-only label/value/unit comparison, not bag-of-numbers overlap."""
    key = lambda item: (item["label"].strip().casefold(), item["unit"].strip().casefold())
    predicted_keys = [key(item) for item in predicted]
    duplicates = sorted({pair for pair in predicted_keys if predicted_keys.count(pair) > 1})
    pred = {key(point): point["value"] for point in predicted}
    gold = {key(point): point["value"] for point in expected}
    mismatched = [{"label_unit": list(pair), "expected": gold[pair], "predicted": pred[pair]} for pair in sorted(gold.keys() & pred.keys()) if not np.isclose(pred[pair], gold[pair], atol=1e-8, rtol=0)]
    missing = sorted(gold.keys()-pred.keys())
    unexpected = sorted(pred.keys()-gold.keys())
    return {"ok": not (duplicates or mismatched or missing or unexpected), "mismatched": mismatched, "missing": missing, "unexpected": unexpected, "duplicates": duplicates}


def feature_fingerprint(record, model_id, parser_version, preprocessing):
    """Cache identity changes whenever source or feature-producing inputs change."""
    components = {"source_sha256": hashlib.sha256(Path(record.path).read_bytes()).hexdigest(), "model_id": model_id, "parser_version": parser_version, "preprocessing": preprocessing}
    serialized = json.dumps(components, sort_keys=True).encode()
    return {**components, "feature_key": hashlib.sha256(serialized).hexdigest()}


def read_pdf(path, output_dir=None):
    """Extract text and render pages with content-addressed snapshot identities.

    The stem plus source hash identifies this document version, so two unrelated
    files named report.pdf coexist. A corrected file gets a new snapshot ID. A
    production catalog can separately retain a logical document ID across those
    versions; this small helper does not infer that relationship from filenames.
    """
    import pypdfium2 as pdfium
    path = Path(path)
    source_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    # Two PDFs (or two versions of one PDF) must never overwrite prior evidence.
    snapshot_id = f"{path.stem}-{source_sha256[:16]}"
    output_dir = Path(output_dir or CORPUS / "pdf_pages") / snapshot_id
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    document = pdfium.PdfDocument(str(path))
    try:
        for page_number in range(len(document)):
            page = document[page_number]
            text_page = page.get_textpage()
            text = text_page.get_text_range()
            rendered = page.render(scale=1.5)
            destination = output_dir / f"page_{page_number+1:02d}.png"
            rendered.to_pil().save(destination)
            parent_id = snapshot_id
            identifier = f"{parent_id}#page={page_number+1}"
            records.append(Record(identifier, "pdf_page", str(destination), text, identifier, parent_id, {"page": page_number+1, "original_path": str(path), "source_sha256": source_sha256, "extraction": "native PDF text; rendering always preserved"}))
            rendered.close()
            text_page.close()
            page.close()
    finally:
        document.close()
    return records


def sample_keyframes(path, every_seconds=2.0, output_dir=None):
    """Read actual frames from a constant-frame-rate video, never invent fallbacks.

    Times use frame_number/fps for this CFR fixture. General variable-frame-rate
    media needs decoded presentation timestamps, which this helper does not read.
    """
    import cv2
    if every_seconds <= 0:
        raise ValueError("Sampling interval must be positive")
    output_dir = Path(output_dir or CORPUS / "keyframes")
    output_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    frames = []
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("Cannot establish frame timestamps: invalid FPS")
        frame_number, next_sample = 0, 0.0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            timestamp = frame_number / fps
            if timestamp + 1e-8 >= next_sample:
                destination = output_dir / f"frame_{frame_number:05d}.png"
                if not cv2.imwrite(str(destination), frame):
                    raise ValueError(f"Cannot write sampled frame {destination}")
                frames.append({"id": f"frame_{frame_number:05d}", "time": timestamp, "path": str(destination)})
                next_sample += every_seconds
            frame_number += 1
    finally:
        capture.release()
    if not frames:
        raise ValueError("Video decoded no frames; do not claim a visual search ran")
    return frames


def build_video_records(segments, frames):
    """Align half-open windows for this single solar_lab_review teaching video.

    Out-of-window nearest frames are deliberately not attached: silent gaps
    remain gaps instead of acquiring unrelated visual evidence.
    """
    transcript_records, frame_records = [], []
    for segment in segments:
        start, end = segment["start"], segment["end"]
        if start >= end:
            raise ValueError("Transcript windows need start < end")
        source = "solar_lab_review#"+segment["id"]
        meta = {"start": start, "end": end, "transcript_provenance": "hand-authored companion; silent video"}
        transcript_records.append(Record(segment["id"], "video_text", str(CORPUS/"solar_lab_review.avi"), segment["text"], source, "solar_lab_review", meta))
        for frame in frames:
            if start <= frame["time"] < end:
                frame_records.append(Record(frame["id"], "video_frame", frame["path"], "", source, "solar_lab_review", {**meta, "sample_time": frame["time"]}))
    if len({r.id for r in frame_records}) != len(frame_records):
        raise ValueError("Overlapping windows assigned a frame more than once")
    return transcript_records, frame_records


def timestamp(seconds):
    if not np.isfinite(seconds) or seconds < 0:
        raise ValueError("Timestamps must be finite and nonnegative")
    tenths = int(round(float(seconds) * 10))
    minutes, remainder = divmod(tenths, 600)
    return f"{minutes:02d}:{remainder / 10:04.1f}"


def evidence_bundle(hits, transcript_records=(), frame_records=(), max_images=3):
    """Attach source-local evidence only after retrieval, never from answer keys.

    Video retrieval ranks segments; its evidence can include the actual sampled
    frame and transcript from that selected segment. Unretrieved segments never
    enter the bundle. Prefer the actual retrieved frame; only a transcript hit
    permits a labelled representative-frame fallback within the selected window.
    """
    bundle, images_used, seen = [], 0, set()
    for hit in hits:
        source = hit.get("source_id", hit["id"])
        if source in seen:
            continue
        seen.add(source)
        item = {"source_id": source, "text": hit.get("text", ""), "images": [], "meta": hit.get("meta", {})}
        if hit["modality"].startswith("video_"):
            text_matches = [r for r in transcript_records if r.source_id == source]
            visual_matches = sorted([r for r in frame_records if r.source_id == source], key=lambda r: r.meta["sample_time"])
            item["text"] = "\n".join(r.text for r in text_matches)
            # Fusion retains evidence in first-seen rank order. A directly
            # retrieved frame precedes later evidence for the same segment.
            candidates = [hit] + hit.get("evidence", [])
            retrieved_visual = next((e for e in candidates
                if e.get("modality") == "video_frame"
                and e.get("source_id", e["id"]) == source
                and e.get("path") and "sample_time" in e.get("meta", {})), None)
            selected_frame = None
            if retrieved_visual:
                selected_frame = {"path": retrieved_visual["path"], "sample_time": retrieved_visual["meta"]["sample_time"], "selection": "retrieved_frame"}
            elif visual_matches and any(e.get("modality") == "video_text" for e in candidates):
                frame = visual_matches[0]
                selected_frame = {"path": frame.path, "sample_time": frame.meta["sample_time"], "selection": "representative_for_retrieved_transcript"}
            if selected_frame and images_used < max_images:
                item["images"] = [selected_frame]
                images_used += 1
        elif hit["modality"] in {"image", "pdf_page"} and images_used < max_images:
            item["images"] = [{"path": hit["path"]}]
            images_used += 1
        bundle.append(item)
    return bundle
