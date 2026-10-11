# HotpotQA RAG evaluation lesson

Open [index.html](index.html) directly in a browser, or use the portable [34-page PDF](hotpotqa_rag_evaluation.pdf). The 34-slide lesson uses local HTML, CSS, JavaScript, and 15 original SVG diagrams, with no CDN or server dependency. The lesson takes roughly 60–90 minutes with the companion notebook exercises.

Click any diagram to open its full-size SVG in a new tab.

Use **Previous / Next** or the arrow keys to navigate. **Jump to a slide** opens any topic. **Explain** shows detailed teaching notes. **Read all** shows the continuous lesson, and **Print** produces one slide per page. The PDF contains slide content and source links; use the HTML explanations for detailed teaching notes. On small screens the content stacks vertically and scrolls. Without JavaScript the entire lesson remains readable, including expandable explanations. Each slide has a stable URL fragment.

The deck explains HotpotQA’s raw JSON through a complete invented example and an actual development example. It distinguishes distractor retrieval from fullwiki, retrieval coverage from predicted supporting facts, and scripted graph demonstrations from genuine model evaluations. Worked score examples are illustrative calculations, not benchmark performance claims.

## Teaching sequence

1. Slides 1–6: RAG, HotpotQA, bridge and comparison questions, and candidate scope.
2. Slides 7–12: complete synthetic JSON, actual raw record, original sentence IDs, and label separation.
3. Slides 13–22: experiment protocol, local retrieval, LangGraph routing, generation, and backend modes.
4. Slides 23–29: retrieval diagnostics, answer/SP/joint scores, official export, and paired comparisons.
5. Slides 30–34: failure analysis, reproducibility, limitations, exercises, and reporting.

Sources appear on the slides where they support factual claims. HotpotQA data is credited to Yang et al. (2018) under CC BY-SA 4.0. The original and reconstructed dataset records live in `../data/`, with provenance and licensing information there. The SVG diagrams are newly authored teaching illustrations. Synthetic entity names and numeric metric examples are explicitly labeled.

## Editing and validation

Edit [build_deck.py](build_deck.py) for slide text, notes, and original SVG source. Edit [slides.css](slides.css) for layout and [slides.js](slides.js) for controls. Then rebuild:

```bash
python3 build_deck.py
```

The generator requires `../data/toy_example.json` and checks that the complete teaching JSON shown on slide 7 matches the file exactly. Avoid editing the generated HTML or SVG files directly, since rebuilding replaces them.

Run the browser validator with a Python environment containing Playwright:

```bash
python validate_slides.py
```

The validator opens the deck through `file://` in Chrome or Chromium, checks every slide at desktop and mobile widths, exercises navigation and reading mode, and checks no-JavaScript readability. Review screenshots in its reported temporary output directory. The validator is an authoring tool; learners do not need Playwright to open the deck.
