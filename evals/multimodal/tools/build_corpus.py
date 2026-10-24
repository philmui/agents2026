"""Create an original fictional school solar-lab corpus. No internet or model calls.

The reference parses are hand-authored teaching fixtures. The answer key is kept
separately and must never be passed to a parser, embedder, or answer generator.
Run from this module directory: uv run python tools/build_corpus.py
"""
from pathlib import Path
import json
import math

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets" / "corpus"
COLORS = {"orange": "#d86a28", "purple": "#8664c5", "green": "#3a946e", "teal": "#268f99"}


def font(size, bold=False):
    candidates = [
        f"/System/Library/Fonts/Supplemental/Arial{' Bold' if bold else ''}.ttf",
        f"/usr/share/fonts/truetype/dejavu/DejaVuSans{'-Bold' if bold else ''}.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


def chart(filename, title, subtitle, labels, values, unit, color):
    image = Image.new("RGB", (1000, 650), "#fcfbf8")
    draw = ImageDraw.Draw(image)
    draw.text((60, 35), "CEDAR SCHOOL / SOLAR LAB", font=font(17, True), fill="#636363")
    draw.text((60, 80), title, font=font(36, True), fill="#242424")
    draw.text((60, 135), subtitle, font=font(21), fill="#525252")
    left, right, top, bottom = 110, 930, 215, 530
    # Four equal ticks with exact integer labels; avoid rounded axis values.
    max_y = math.ceil(max(values) * 1.3 / 4) * 4
    for tick in range(5):
        value = max_y * tick / 4
        y = bottom - (bottom - top) * tick / 4
        draw.line((left, y, right, y), fill="#dddcd7", width=1)
        draw.text((48, y - 10), f"{value:.0f}", font=font(16), fill="#666666")
    width = (right-left) / len(labels)
    for index, (label, value) in enumerate(zip(labels, values)):
        x = left + width * (index + 0.5)
        y = bottom - (bottom-top) * value/max_y
        draw.rectangle((x-65, y, x+65, bottom), fill=COLORS[color])
        draw.text((x, y-33), f"{value} {unit}", font=font(22, True), fill="#242424", anchor="mm")
        draw.text((x, bottom+28), label, font=font(22), fill="#242424", anchor="mm")
    draw.text((60, 605), "Fictional teaching data • each label belongs to its own bar", font=font(16), fill="#666666")
    path = OUT / filename
    image.save(path)
    return path


def screenshot():
    image = Image.new("RGB", (1000, 650), "#fcfbf8")
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((45, 45, 955, 605), radius=20, fill="white", outline="#d6d5d0", width=2)
    draw.rounded_rectangle((45, 45, 955, 160), radius=20, fill=COLORS["teal"])
    draw.text((85, 80), "Solar controller · inspection screen", font=font(32, True), fill="white")
    draw.text((85, 200), "Unit: rooftop controller A", font=font(27), fill="#333333")
    draw.text((85, 275), "STATUS", font=font(18, True), fill="#666666")
    draw.text((85, 315), "E17 — ventilation blocked", font=font(36, True), fill="#222222")
    draw.text((85, 410), "Action: clear the vent before restarting.", font=font(25), fill="#333333")
    draw.text((85, 485), "Recorded: 14:20 • Tuesday inspection", font=font(22), fill="#555555")
    draw.text((85, 550), "Fictional interface · no live equipment", font=font(17), fill="#777777")
    image.save(OUT / "screenshot_controller.png")


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    charts = [
        ("chart_energy", "Solar energy by month", "Energy delivered by the school rooftop panels", ["April", "May", "June"], [42, 57, 51], "kWh", "orange"),
        ("chart_temperature", "Panel temperature by location", "Surface temperature during the same midday inspection", ["Roof", "Garden", "Shade"], [46, 34, 27], "°C", "purple"),
        ("chart_water", "Cleaning water by method", "Water needed to clean one demonstration panel", ["Hose", "Bucket", "Mist"], [18, 8, 3], "L", "green"),
    ]
    parses, key, manifest = {}, {}, []
    for identifier, title, subtitle, labels, values, unit, color in charts:
        chart(identifier+".png", title, subtitle, labels, values, unit, color)
        parses[identifier] = {
            "kind": "bar chart", "title": title,
            "summary": f"{subtitle}. A {color} bar chart.",
            "data_points": [{"label": label, "value": value, "unit": unit} for label, value in zip(labels, values)],
            "provenance": "hand-authored reference fixture; not a model prediction",
        }
        key[identifier] = {"data_points": parses[identifier]["data_points"]}
        manifest.append({"id": identifier, "modality": "image", "path": identifier+".png", "source_id": identifier, "parent_id": identifier})
    screenshot()
    parses["screenshot_controller"] = {
        "kind": "screenshot", "title": "Solar controller inspection screen",
        "summary": "A teal status header. Rooftop controller A shows E17: ventilation blocked. Clear the vent before restarting. Tuesday inspection recorded at 14:20.",
        "data_points": [], "provenance": "hand-authored reference fixture; not a model prediction",
    }
    key["screenshot_controller"] = {"code": "E17", "meaning": "ventilation blocked", "action": "clear the vent before restarting"}
    manifest.append({"id": "screenshot_controller", "modality": "image", "path": "screenshot_controller.png", "source_id": "screenshot_controller", "parent_id": "screenshot_controller"})
    memos = {
        "memo_safety": "# Solar lab safety\nStudents must stay off the roof. Only trained staff inspect rooftop solar panels. If the controller reports a ventilation warning, stop the demonstration and ask a teacher to inspect the vent. Do not restart equipment yourself.\n",
        "memo_schedule": "# Solar lab demonstration schedule\nThe rooftop solar energy review happens on Thursday in the science classroom. Bring the monthly energy chart to discuss how energy changes across spring and summer. Weather and maintenance can both affect production; a bar chart alone does not prove the cause.\n",
        "memo_cleaning": "# Panel cleaning trial\nThe class compared hose, bucket, and mist cleaning methods. The green water-use chart records water consumed. Choose a method only after checking cleaning quality too. Using less water does not automatically mean the panel becomes equally clean.\n",
    }
    for identifier, body in memos.items():
        (OUT / (identifier+".md")).write_text(body, encoding="utf-8")
        manifest.append({"id": identifier, "modality": "text", "path": identifier+".md", "source_id": identifier, "parent_id": identifier})
    # A native text page and a raster chart page expose the PDF extraction boundary.
    pdf = canvas.Canvas(str(OUT / "solar_lab_report.pdf"), pagesize=letter, invariant=1)
    pdf.setTitle("Cedar School solar lab report — fictional teaching corpus")
    pdf.setFont("Helvetica-Bold", 22)
    pdf.drawString(55, 725, "Cedar School solar lab report")
    pdf.setFont("Helvetica", 13)
    for i, line in enumerate([
        "Page 1: native PDF text", "The school reviews solar energy each Thursday.",
        "Students compare monthly energy in the chart on page 2.",
        "A chart can show a pattern without proving its cause.",
        "Report version: classroom demonstration.",
    ]):
        pdf.drawString(55, 665-i*34, line)
    pdf.showPage()
    pdf.drawImage(str(OUT / "chart_energy.png"), 30, 220, width=552, height=359)
    pdf.showPage()
    pdf.save()
    # Silent 12 s, 5 fps MJPG video: three four-second scenes. No ASR claim.
    path = OUT / "solar_lab_review.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 5.0, (1000, 650))
    if not writer.isOpened():
        raise RuntimeError("OpenCV cannot write MJPG; corpus creation stopped rather than inventing a video.")
    scenes = ["chart_energy.png", "screenshot_controller.png", "chart_temperature.png"]
    try:
        for filename in scenes:
            frame = cv2.imread(str(OUT/filename))
            for _ in range(20):
                writer.write(frame)
    finally:
        writer.release()
    transcript = [
        {"id": "seg_000", "start": 0.0, "end": 4.0, "text": "Here is the monthly rooftop solar energy chart. Compare the spring and summer bars."},
        {"id": "seg_001", "start": 4.0, "end": 8.0, "text": "The controller warning appeared during the inspection. The teacher cleared the blocked vent before restarting."},
        {"id": "seg_002", "start": 8.0, "end": 12.0, "text": "We finish by comparing panel temperatures at the roof, garden, and shade locations."},
    ]
    (OUT / "transcript.json").write_text(json.dumps({"provenance": "hand-authored companion transcript; video is silent; not ASR output", "segments": transcript}, indent=2), encoding="utf-8")
    for name, obj in [("manifest.json", manifest), ("reference_parses.json", parses), ("answer_key.json", key)]:
        (OUT/name).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT/"README.md").write_text("# Original classroom corpus\n\nAll people, records, measurements, and interfaces are fictional. Generated by `tools/build_corpus.py`; no external media or personal data.\n\nThe three charts store their measurements in pixels. The raw memos contain no chart measurements. `reference_parses.json` is a hand-authored ingestion fixture, not model output. `answer_key.json` is evaluation-only ground truth. Keeping those separate prevents evaluating a model against its own prediction.\n\n`solar_lab_report.pdf` has native text on page 1 and a raster image on page 2. `solar_lab_review.avi` is a silent, 12-second MJPG video; `transcript.json` is a hand-authored companion transcript, not an automatic transcription.\n", encoding="utf-8")
    print(f"Created {len(manifest)} source records, PDF, video, and labelled fixtures in {OUT}")


if __name__ == "__main__":
    build()
