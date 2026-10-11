# Adversarial review of the classroom deck

This revision examines the lesson from four perspectives: a student meeting the terminology, a reader interpreting diagrams without narration, a presenter using the controls, and a reviewer trying unusual inputs. It combines close visual inspection with browser checks. It is an authored adversarial review, rather than a study with student participants.

The frontend-slides skill supplied fixed-stage, density, editing, and responsive checks. The guizang-ppt-skill supplied presentation rhythm, current/next previews, overview navigation, and visible-boundary checks. The user's pastel preference and need for labeled, reusable SVG illustrations guide this existing-deck enhancement; it does not adopt a locked Swiss template or its template-specific validator.

## Findings and corrections

| Issue challenged | Why it matters | Correction |
| --- | --- | --- |
| Generic boxes concealed important mechanics | Students could memorize arrows without understanding the computation | RNN states, CNN windows, numerical naive Bayes scores, a logistic curve, and an agent feedback loop now explain the operation |
| Attention grids lacked axes and a worked row | A mask could be mistaken for measured attention strength | Both axes are labeled; the bank row is outlined; check marks and crosses show permitted and blocked positions |
| Counts had the same emphasis as labels | The calculation was hard to find | Large counts sit in numbered vocabulary positions, with the resulting vector written below |
| TF-IDF lacked a document-frequency example | Rarity sounded like an unexplained property of a word | Six training documents compare a ubiquitous word with a word present in two documents |
| Probability, confidence, and coverage could be conflated | A correct-looking number could justify an unsuitable routing policy | Worked examples show normalization, the even-split baseline, confidence, and coverage; routing uses an explicitly illustrative threshold |
| Some meanings depended on color alone | Grayscale or color differences could erase the distinction | Attention uses ✓/×, coverage uses A/R, calibration marks correct/incorrect, and probabilities have numeric labels |
| Coincident merge arrows formed a dark cluster | The direction of answers became ambiguous | Parallel and multi-agent returns join on a bus before one arrow enters the combining step |
| A routing label crossed its arrow | The branch was harder to read | The no label is clear of the connector |
| Cramped caption spacing | Wrapped explanations were uncomfortable to read | Line spacing and node heights were increased; text is checked against its node bounds |
| Cover graphics shrank within a wide panel | The example became too small | A viewBox matching the illustration's content fills the panel width |
| Readers could lose their place among 52 slides | Chapter jumping could not find a particular concept | A searchable overview supports title, chapter, and number, with empty-results feedback |
| Small controls and hidden editing | Touch and keyboard users lacked a reliable path to tools | Navigation has 44-pixel targets and a visible More menu |
| Notes covered diagrams | Readers could not compare the explanation with its example | Wide-screen notes reserve their own space beside the uniformly scaled stage |
| Invalid URL fragments hid every slide | A malformed link could produce a blank deck | Invalid fragments fall back to slide 1; out-of-range values are clamped |
| Space navigated while activating a focused button | Keyboard behavior depended on focus history | Focused buttons retain native Space/Enter activation |
| Escape in search cleared the query instead of closing | The advertised close shortcut was inconsistent | Dialog Escape handling closes directly and prevents the same keystroke reopening the overview |
| Edit storage used positional fields | Reordering or another file could inherit unrelated edits | Storage uses revision, path, stable slide ID, and field name; reset and export are exposed |
| Presenter preview extended below controls | The upcoming slide was clipped despite its correct ratio | Both previews share a measured height budget and remain fully visible at tested desktop sizes |
| Docs referenced a notebook filename absent from the workspace | Instructions led to a missing file | References and the builder use the existing system1_tutorial.ipynb filename |

The revised prose defines RNN, CNN, classification head, and [CLS] where students encounter them. It distinguishes a two-class logistic example from a three-option Choice answer. Jev's public interface remains separate from its undisclosed architecture and training details, and invented examples remain separate from measured results.

## Verification evidence

The [browser checker](../../scripts/check_slides.py) visits all 52 slides at **1280×720, 390×844, 844×390, and 1920×1080**, for **208 slide/viewport checks**. It checks stage proportions, canvas/navigation separation, heading/explanation spacing, diagram/takeaway spacing, SVG bounds, text collisions, node containment, control sizes, unique IDs, and attribution.

Interaction checks cover keyboard navigation, chapter selection, overview search and empty results, dialog closing, native button activation, notes layout, malformed links, edit persistence/reset, a downloaded standalone HTML copy, presenter synchronization in both directions, current/next aspect ratios and visibility, and the final-slide state. Ordinary motion reaches a readable final state; reduced-motion rendering is static. The run records no browser exceptions or remote resource requests. The recorded result is in [review-report.json](review-report.json).

All 52 standalone SVG files also passed viewBox and node-containment checks when opened independently with the browser’s fallback sans-serif font. All 52 slide screenshots were reviewed in contact sheets, with detailed inspection of the cover, count vector, RNN, CNN, attention mask, confidence example, routing branches, overview, reading panel, presenter previews, and phone layout. This visual pass caught merge-arrow and presenter-clipping issues beyond the initial text-boundary checks.

The regenerated PDF has **52 pages**, vector diagrams, and **@mui-group on every page**. The notebook's 20 shared SVG attachments were synchronized and validated while preserving all 29 code cells' retained outputs. This design review did not call the hosted Jev API or rerun unchanged notebook calculations.

## Practical limits and next classroom check

The fixed slide canvas remains too small for comfortable close reading on a portrait phone. Landscape viewing or the PDF's zoom gives the content more room. A classroom should also check projection contrast and legibility from the back of the room. Browser geometry cannot establish how readily students learn the concepts.

Ask students to reconstruct the count vector, explain which tokens bank can read under each mask, and distinguish probability, confidence, and coverage without consulting notes. Their answers can reveal where another worked example would help. Custom browser edits need a fresh visual check because longer text can exceed the authored space.

The lightweight presenter view synchronizes two open windows and reports whether the audience tab remains open. It does not inspect projector hardware or provide a timed rehearsal studio. Reopen presentation windows from the audience tab when starting another session.
