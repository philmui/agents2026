# Tutorial slides

[Download the PowerPoint deck](outputs/pptx/multimodal-retrieval.pptx): 42 widescreen
slides with editable text, diagrams, four data charts, and complete speaker notes.

Open [slides.html](../slides.html) directly in a browser. The 42-slide deck works
from `file://` and needs no server, network connection, or downloaded fonts.

- Use the arrow buttons or **← / →** keys to navigate.
- **Home / End** jump to the first or last slide.
- **N** or **Speaker notes** opens the full explanation and notebook link.
- The slide selector jumps to a specific topic.
- **Open static slide** opens the current 1600 × 900 slide by itself.

The [static index](index.html) links every slide and includes all explanations
without JavaScript. [speaker_notes.md](speaker_notes.md) is the plain Markdown
version. [manifest.json](manifest.json) records source anchors, layout roles,
code excerpts, and notebook connections.

Standalone slides live in [outputs/html](outputs/html). Review previews and
contact sheets live in [assets/slide_previews](../assets/slide_previews), including
[slides 1–14](../assets/slide_previews/contact_1.png),
[slides 15–28](../assets/slide_previews/contact_2.png), and
[slides 29–42](../assets/slide_previews/contact_3.png).

## Rebuild

From `evals/multimodal`:

```sh
uv run python tools/build_slides.py
```

The generator uses only the Python standard library, so `python3` also works.
It regenerates the HTML, static index, notes, and manifest. It does not need
the model dependencies or API credentials.

## Rebuild PowerPoint

The [PowerPoint exporter](../tools/export_powerpoint.mjs) reads the rendered HTML
positions and creates native PowerPoint text and shapes. Solar-energy charts
include editable data workbooks. Speaker notes come from `manifest.json`.
The export uses Arial and Courier New for portability.

Install the optional Node dependencies outside the lesson's Python environment:

```sh
npm install --prefix /tmp/multimodal-pptx-build pptxgenjs@4.0.1 playwright@1.64.1
/tmp/multimodal-pptx-build/node_modules/.bin/playwright install chromium
node tools/export_powerpoint.mjs /tmp/multimodal-pptx-build/node_modules
```

Alternatively, set `CHROMIUM_EXECUTABLE` to an installed Chromium or Chrome
executable. An optional second argument after the package directory changes the
output `.pptx` path. After updating the lesson, rebuild the HTML first, then
rebuild PowerPoint and inspect its rendered slides.

## Optional browser review

Rendering is a maintenance step, separate from running the lesson. It requires
Node.js, the Playwright Node package, and a compatible installed Chromium
browser. It does not come from the module's Python environment. One setup is:

```sh
# Install these in a separate scratch directory.
npm install playwright
npx playwright install chromium

# Then return to evals/multimodal.
node slides/render_review.mjs /absolute/path/to/node_modules/playwright
```

If Chromium is installed elsewhere, pass its executable path as a third
argument after the Playwright package path. The review saves slide PNGs and
contact sheets in `assets/slide_previews`, and records geometry and shell checks
in [render_review.json](render_review.json).

[REVIEW.md](REVIEW.md) documents the actual review, adversarial findings, root
causes, and repairs. The slides contain illustrative math and explicitly labeled
fixture limitations; they do not claim unrun model benchmarks.

© mui-group
