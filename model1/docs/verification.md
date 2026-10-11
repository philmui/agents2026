# Verification and limits

The tutorial's verification checks the deliverables and executable teaching mechanisms. It does not establish the quality of the hosted Jev service.

## Notebook

The notebook was validated as nbformat 4 and its Python cells were compiled. Its offline path was executed in a fresh Python 3.14 kernel, including the optional LangGraph workflow with fixture responses. Retained outputs come from those executions.

The checks cover training-only vocabulary fitting, toy naive Bayes predictions, probability normalization, Choice confidence, Score expectations, response parsing, routing, graph branch equivalence, Brier calculations, and separate development/test threshold application. Figures are embedded SVG attachments or computed vector outputs.

All distributed live flags are false. No hosted Jev request, LangSmith upload, or remote agent execution was needed for validation. API and adapter examples were checked against current documentation and local package interfaces. The official SDK and LangChain adapter also passed local mock-transport checks of request serialization and typed response parsing. This validates their construction, not an actual provider response.

## Slides

The revised pastel deck passes 208 geometry checks: all 52 slides at 1280×720, 390×844, 844×390, and 1920×1080. Checks cover the fixed 16:9 canvas, one active slide, slide/control separation, text and SVG bounds, node containment, labels, and attribution. All slide screenshots were reviewed, with detailed inspection of important diagrams and interface states. All 52 standalone SVG assets also passed viewBox and node-containment checks with the fallback browser font.

Interaction checks cover overview search, keyboard and chapter navigation, notes, invalid links, edit persistence/reset/export, and presenter synchronization in both directions. Current and next previews remain visible at 1440×900 and 1280×720. No browser errors or remote resource requests were recorded. See [the review](slides/review.md) and [recorded result](slides/review-report.json).

The HTML embeds both fonts and all 52 diagrams. Every slide credits @mui-group. The regenerated vector PDF has 52 pages and attribution on every page. The notebook's 20 shared illustrations were synchronized while preserving its 29 code cells' outputs; unchanged computations were not rerun for this design revision.

## Reproduce the checks

Rebuild the artifacts with the two scripts described in README.md. To execute the notebook with base notebook requirements installed:

```bash
python -m jupyter nbconvert --to notebook --execute --inplace notebooks/system1_tutorial.ipynb
```

Use a kernel with LangGraph installed to exercise that optional section. Otherwise the direct Python workflow still executes and the graph cell explicitly reports its skip.

Manual visual review should include the title, word-count vector, attention mask, fractional Score, confidence formula, graph branches, and early-study diagram. Print-to-PDF removes animations and interactive notes while preserving the final slide content.

Reproduce the slide review with `scripts/check_slides.py`; installation and PDF regeneration are documented in [the slide guide](slides/README.md#reproduce-the-review-and-pdf).
