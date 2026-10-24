# Slide review and maintenance

The presentation has 42 standalone, static 1600 × 900 HTML slides. Open
[the presentation shell](../slides.html) for navigation and notes, or use
[the static index](index.html) without JavaScript. All figures are inline SVG;
there are no web fonts, external image requests, or slide-level scripts.
Every slide carries `© mui-group`.

The generator is [build_slides.py](../tools/build_slides.py). It uses only the
Python standard library. Regenerate from the module directory with:

```sh
python tools/build_slides.py
```

[manifest.json](manifest.json) records each slide's content role, layout,
notebook, reference-notebook cell anchors, code excerpt, and full explanation.
[speaker_notes.md](speaker_notes.md) provides the explanations separately.
Code snippets are shortened architecture examples; the notebook cells contain
the runnable versions. The seeking example on slide 36 is explicitly an
alternative to the notebook's sequential constant-frame-rate sampler.

## Review performed

All 42 slides were rendered in Chromium at their native 1600 × 900 size. PNGs and
contact sheets are stored in [assets/slide_previews](../assets/slide_previews).
The three contact sheets were inspected for hierarchy, visual pacing, contrast,
label placement, and diagram meaning. Individual views were used for the shared
space, rank-fusion example, video sampling, and notes interface after revisions.
The palette follows the user's white, charcoal, coral, and teal constraints;
the user's sans-serif requirement overrides the style skill's serif option.

The programmatic review in [render_review.mjs](render_review.mjs) checks rendered
text bounds, text collisions, projected SVG-label sizes, and code overflow.
The final [render_review.json](render_review.json) records **42 rendered slides
and zero detected geometry issues**. These checks support visual review; they
do not establish mathematical or pedagogical correctness by themselves.

The shell checks passed for next/previous controls, Home/End shortcuts, keyboard
navigation after clicking the slide, speaker-note toggling, a 390-pixel mobile
viewport without horizontal overflow, and the no-JavaScript fallback.

To repeat the optional browser review, install Playwright in a separate Node
environment and run:

```sh
node slides/render_review.mjs /absolute/path/to/playwright
# Optional third argument: a compatible Chromium executable path.
```

## Adversarial findings and repairs

| Finding | Root cause | Repair and verification |
|---|---|---|
| Framework diagram labels projected below 16 pixels | A full-width SVG was reused in a narrow code column | Rebuilt it as a 700-pixel two-branch diagram; native render now passes |
| Source-record title crossed field labels | A centered box-label helper was used for a record header | Positioned the header independently; overlap check now passes |
| Fusion example could imply agreement across disjoint IDs | Caption-overlap assumption appeared only in notes | Visible `Caption variant` label plus explicit notes about disjoint stores |
| Video endpoint looked like an observed frame | Identical filled markers represented samples and the exclusive end | Made the endpoint hollow and labeled `12 s end` |
| Some source-code anchors pointed to neighboring cells | Concept ranges were carried into exact-code attribution | Checked zero-based cell summaries against the original notebook and corrected anchors |
| JSON example used unquoted keys | Prose shorthand conflicted with a schema-validation lesson | Replaced it with syntactically valid quoted JSON |
| Gateway alias attribution was imprecise | The user-requested alias was described as literal reference code | Identified it as course configuration, with a separate embedding model |
| Clicking an iframe could consume keyboard focus | `file://` isolation can prevent cross-frame listeners | Disabled iframe pointer events and made the parent stage focusable; tested post-click arrows |
| JavaScript-disabled shell would show no lesson | Navigation was entirely scripted | Added a static index with all notes and a visible no-JavaScript link |

The independent adversarial content review was performed by the guide/assets
agent. No strategy benchmark, API response, or model quality measurement is
invented in the deck. Illustrative arithmetic and two-dimensional projections
are labeled. Fixture captions and color features are kept separate from live
vision-model parsing and pretrained CLIP results.
