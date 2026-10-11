# Presenting From Bag-of-Words to Jev

The [deck](index.html) has 52 slides, each with an original SVG illustration and explanatory teaching notes. A paper background, powder blue, mint, peach, and lavender give the lesson a calm pastel palette. Embedded Bricolage Grotesque headings and Source Sans 3 body text provide a consistent hierarchy. Every slide credits **@mui-group**, the Mui Research Group at ASDRP.

Open `index.html` directly in a browser. No installation, network connection, or server is required. Slides retain a 16:9 canvas and scale uniformly; phones show the same layout with letterboxing. The navigation controls occupy a separate area below the canvas.

A [52-page PDF copy](from-bag-of-words-to-jev.pdf) is available for sharing and printing. It preserves vector graphics and attribution; animations, editing, and interactive speaker notes belong to the HTML version. Regenerate the PDF with the browser's print dialog after editing or rebuilding the deck.

## Navigation

| Control | Action |
| --- | --- |
| Right/Down arrow, Space, Page Down | Advance |
| Left/Up arrow, Page Up | Go back |
| Home / End | First / last slide |
| Horizontal swipe or mouse wheel | Navigate |
| Chapter menu | Jump to the first slide in a chapter |
| Slide count / Escape | Open the searchable slide overview |
| N / Notes | Read the explanation and source link |
| P | Open the presenter window |
| B | Toggle a static display |
| ? | Open help and shortcuts |
| More | Full screen, editing, HTML export, PDF printing, and help |
| E / upper-left edit control | Edit the heading, explanation, and takeaway |
| Escape | Close notes and finish editing |
| Ctrl/Cmd+S while editing / More → Save HTML copy | Download an edited HTML copy |
| More → Print / save PDF | Open the browser print dialog; one slide per page |

Escape first closes an open dialog or notes panel, or finishes editing. When none is open, it shows the overview. Space activates a focused button. Modal dialogs contain keyboard focus and have explicit close buttons. On narrow screens, find chapters through the searchable overview; rotate to landscape for larger slide text. Navigation targets remain at least 44 pixels wide and tall. Motion respects the device's reduced-motion preference; B also provides a manual static mode.

## Present on two screens

Press **P** or choose **More → Open presenter window**. Move the original audience tab onto the projector or shared screen, and keep the presenter window on your own display. It shows the current slide, a smaller next preview, and teaching notes. Navigation synchronizes in both directions while both windows remain open.

Allow a pop-up for this local file if the browser blocks the presenter window. The status describes whether the original audience tab remains open; it does not inspect projector hardware. If you close the audience tab, reopen the deck and launch its presenter window again. Home returns to the start. At the final slide, the next-preview area displays the end of the lesson. Notes scroll within their own panel, while both previews stay above the controls.

## Edit and preserve a copy

Browser edits save in local storage using file path, revision, stable slide ID, and field name. Download an edited copy to preserve it outside that browser. **More → Reset browser edits** restores all editable text saved in the opened file. Keep changes within the existing space and inspect long replacements for overflow. Editable text does not change teaching notes or the source builder, so update those separately when revising an explanation. Rebuilding generated files does not merge browser edits. A downloaded copy works offline and opens at the beginning.

For an authored revision, edit the lesson and drawing primitives in [build_slides.py](../../scripts/build_slides.py). Diagram refinements, pastel styling, and browser behavior live in [slide_design.py](../../scripts/slide_design.py). Run from the repository root:

```bash
python scripts/build_slides.py
```

The builder writes `index.html`, `deck.json`, and the SVG assets. `deck.json` is an inspectable generated outline with explanations, notes, and citations; it is not the builder's input. Theme colors are defined in the builder's CSS and the generated HTML's `:root`. Illustration labels belong to the SVG drawing functions.

If shared illustrations change, regenerate the notebook as well:

```bash
python scripts/build_notebook.py
```

That replaces notebook outputs; execute the notebook afterward. The checked-in font files and [viewport-base.css](viewport-base.css) are sufficient for offline rebuilding. Font licensing is recorded in [the font license directory](assets/fonts).

## Reproduce the review and PDF

The [adversarial review](review.md) records the issues, fixes, and practical limits. Install the optional review tools in a separate Python environment:

```bash
python -m pip install -r scripts/requirements-review.txt
python -m playwright install chromium
python scripts/check_slides.py --screenshots /tmp/system-one-slide-review --pdf
```

Use `--browser` to select an installed Chrome binary instead of downloading Chromium. On macOS:

```bash
python scripts/check_slides.py \
  --browser "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
  --screenshots /tmp/system-one-slide-review --pdf
```

The checks visit every slide at four viewport sizes, exercise controls and presenter synchronization, and verify PDF page count and attribution. Screenshot output belongs in a temporary directory. You can also use **More → Print / save PDF**; select landscape, remove browser headers and footers, and include background graphics. See [review-report.json](review-report.json) for the recorded result.

## Teaching sequence

| Slides | Topic |
| --- | --- |
| 1–6 | The classification task, labels, representations, and historical map |
| 7–15 | Tokens, word counts, naive Bayes, weighted classifiers, TF-IDF, and data splits |
| 16–22 | Embeddings, RNNs, LSTMs, CNNs, pooling, and transfer learning |
| 23–29 | Attention, encoders, decoders, T5, and choosing an approach |
| 30–38 | System One, Jev's public contract, three primitives, confidence, and calibration |
| 39–44 | Parallel questions, structured state, question design, and routing |
| 45–48 | Agent harnesses, LangGraph, trace evaluation, and the scope of early evidence |
| 49–52 | Open-model experiments, multi-agent coordination, and fair research |

Use [the teaching guide](../teaching-guide.md) for pacing. The deck's occasional numerical examples are explicitly illustrative; they are not Jev measurements.
