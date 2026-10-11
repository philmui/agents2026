# From Bag-of-Words to Jev

A visual, hands-on introduction to language models for text classification, created for STEM high school students by the **Mui Research Group at ASDRP · @mui-group**.

The tutorial starts with a school help desk: how can software decide whether a message needs technical help, billing help, or information? It builds from word counts to contextual language representations, then introduces Jev's System One interface and its use in routed workflows and agent evaluation.

The materials explain features, labels, probabilities, and evaluation before relying on those ideas. They include original SVG diagrams, worked Python examples, exercises, and six research projects involving agents, cooperating agents, and broader machine learning.

## Start with the materials

| Material | Location | Purpose |
| --- | --- | --- |
| Visual tutorial | [52-slide deck](docs/slides/index.html), [PDF copy](docs/slides/from-bag-of-words-to-jev.pdf) | Step-by-step explanations with 52 original vector illustrations and speaker notes in the HTML version |
| Hands-on tutorial | [System One notebook](notebooks/system1_tutorial.ipynb) | 14 stages, 29 executable cells, 20 embedded illustrations, and generated vector charts |
| Documentation index | [docs/README.md](docs/README.md) | A map of the supporting guides |
| Teaching guide | [docs/teaching-guide.md](docs/teaching-guide.md) | Lesson pacing, discussion prompts, and misconceptions |
| Source and evidence notes | [docs/source-notes.md](docs/source-notes.md) | All eleven requested references, API distinctions, and limits of the evidence |
| Research roadmap | [docs/research-roadmap.md](docs/research-roadmap.md) | Hypotheses, controls, measurements, and failure analyses |

The new notebook is independent of the earlier [Jev examples notebook](notebooks/jev_examples.ipynb), which remains available.

## Open the slides

Open `docs/slides/index.html` in a browser. The HTML contains its fonts, vector graphics, styles, and navigation code, so it works offline without a build step or web server. On macOS:

```bash
open docs/slides/index.html
```

For a local HTTP address on any platform:

```bash
python -m http.server 8000
```

Then visit `http://localhost:8000/docs/slides/`. Stop the server with Ctrl+C.

Use the arrow keys or Space to advance, swipe on a touch screen, or choose a chapter. Click the slide count or press **Escape** for a searchable overview. Press **N** for teaching notes, **P** for a presenter window with current and next previews, and **E** for editing. The **More** menu exposes full screen, static display, editing, HTML export, PDF printing, and help. Edits save in browser storage; **More → Save HTML copy** or Ctrl/Cmd+S in edit mode preserves an edited file. On a phone, rotate to landscape for larger text. Every slide footer contains **@mui-group**.

See [the slide guide](docs/slides/README.md) for usage and rebuilding, and [the adversarial review](docs/slides/review.md) for the visualization fixes and 208 browser geometry checks.

## Install and launch the notebook

The standalone tutorial needs **Python 3.10 or later** and the packages in `notebooks/requirements.txt`. It does not require a GPU, a model download, a TypeSafe account, or credentials for its default path. Its core calculations use the Python standard library.

The existing repository project specifies **Python 3.14 or later** in `pyproject.toml`. Choose either of these installation paths; you do not need both.

### Use the existing project with uv

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if needed, then run these commands from `model1/`:

```bash
uv sync
uv pip install --python .venv/bin/python -r notebooks/requirements.txt
uv run --no-sync python -m jupyterlab notebooks/system1_tutorial.ipynb
```

On Windows, use `.venv\Scripts\python.exe` for the `--python` argument. `--no-sync` preserves the extra notebook tools installed in the project environment. If you run `uv sync` again later, reinstall the notebook requirements before launching Jupyter.

### Use a separate environment with Python and pip

On macOS or Linux:

```bash
python3 -m venv .venv-tutorial
source .venv-tutorial/bin/activate
python -m pip install -r notebooks/requirements.txt
python -m jupyterlab notebooks/system1_tutorial.ipynb
```

On Windows PowerShell:

```powershell
py -m venv .venv-tutorial
.venv-tutorial\Scripts\Activate.ps1
python -m pip install -r notebooks/requirements.txt
python -m jupyterlab notebooks/system1_tutorial.ipynb
```

In JupyterLab, choose a Python kernel from the environment you installed into. Select **Kernel → Restart Kernel and Run All Cells**. Read the explanations and outputs in sequence. The vector illustrations are embedded in notebook attachments, so the notebook retains them when copied elsewhere.

## What the offline notebook does

The notebook trains an actual small naive Bayes classifier on twelve synthetic training messages and evaluates it on eight held-out messages. It then uses clearly identified, invented Jev-shaped responses to practice Choice, Noul, Score, response checks, routing, and uncertainty calculations. These fixtures are fixed examples; they cannot classify a new message or measure Jev's quality.

The default run makes **no external API calls**. All four live flags in the setup cell are `False`. The graph section runs with fixture data if LangGraph is installed; otherwise, an equivalent direct Python workflow remains runnable. All handlers return demonstration strings.

The final stages explain an agent trace, calculate probability-quality metrics on controlled synthetic data, compare routing thresholds with separate development and test fixtures, and propose six research projects. Each stage labels whether its evidence is an executed toy calculation, an invented illustration, or a proposed experiment.

## Enable optional live Jev examples

The HTTP example needs a TypeSafe API key but no additional client package. The official SDK, LangChain, and LangGraph examples use optional packages:

```bash
python -m pip install -r notebooks/requirements-live.txt
```

With the uv project environment:

```bash
uv pip install --python .venv/bin/python -r notebooks/requirements-live.txt
```

Create a key in the [TypeSafe console](https://console.typesafe.ai/keys). The [official quickstart](https://docs.typesafe.ai/introduction/quickstart) describes authentication. The direct Python package is `typesafe-sdk`, imported as `typesafe_sdk`; the LangChain adapter is `langchain-typesafe`, imported as `langchain_typesafe`.

Start Jupyter with the key in its environment. This command prompts without displaying the key or writing it into the notebook or a file. Run it with the Python interpreter from the installed environment:

```bash
python - <<'PY'
import getpass
import os
import subprocess
import sys

os.environ["TYPESAFE_API_KEY"] = getpass.getpass("TypeSafe API key: ")
subprocess.run([
    sys.executable, "-m", "jupyterlab",
    "notebooks/system1_tutorial.ipynb",
], check=True)
PY
```

The heredoc command is for macOS/Linux shells. On Windows, the same Python code can be run from a temporary local launcher, or you can set `TYPESAFE_API_KEY` in the process environment before starting Jupyter. Existing Jupyter processes must restart to inherit an updated environment.

Change only the flag for the example you intend to run, rerun the setup cell, then execute that example:

| Flag | Location | Requests per enabled execution |
| --- | --- | --- |
| `RUN_LIVE_HTTP` | Step 8 | One HTTP call using the three tutorial questions |
| `RUN_LIVE_SDK` | Step 8 | One official SDK call using the same three questions |
| `RUN_LIVE_LANGCHAIN` | Step 10 | One LangChain call for urgency |
| `RUN_LIVE_EVALUATION` | Step 12 | Fourteen calls: six development and eight test messages |

Live calls send the synthetic example state to TypeSafe and may incur API charges. They print selected results and returned model information. The notebook never loads the repository's `.env` automatically or prints the API key.

The examples request `jev-1.13.0` by default, the documented version checked on 2026-10-03. To use another version, set `TYPESAFE_MODEL` before launching Jupyter. `jev-latest` is a moving alias; record the actual version reported by the response and reevaluate thresholds when changing versions. See [TypeSafe's models documentation](https://docs.typesafe.ai/models).

## Troubleshooting

| Symptom | Action |
| --- | --- |
| `No module named IPython` or Jupyter | Install the base notebook requirements in the selected kernel's environment. |
| `No module named typesafe_sdk` or `langchain_typesafe` | Install `requirements-live.txt` before enabling that integration. |
| The graph cell is skipped | Install the optional requirements and restart the kernel, or use the direct Python workflow. |
| Missing API key | Start Jupyter from a process with `TYPESAFE_API_KEY` set; changing another terminal does not update a running kernel. |
| HTTP 401/403 | Check the key and account access without displaying the key. |
| HTTP 429 or a timeout | Reduce request rate and retry later; use the official SDK's bounded retry facilities for sustained workloads. |
| A fixture does not change after editing a message | Fixtures are fixed invented responses; use a live call to test new inputs. |
| A numerical score surprises you | Read its probability distribution and rubric. Score averages can be fractional, and Choice confidence differs from top probability. |

## Rebuild and verify the materials

The builders use the standard library and regenerate the authored deck, diagrams, and notebook source:

```bash
python scripts/build_slides.py
python scripts/build_notebook.py
```

Rebuilding the notebook clears its executed outputs. Execute it again to retain fresh results. Rebuilding the deck replaces the generated HTML, SVG illustrations, and JSON outline; keep browser-edited exports separately. The fonts and fixed-stage CSS are already stored locally.

For notebook execution from the command line, after installing the base requirements:

```bash
python -m jupyter nbconvert --to notebook --execute --inplace notebooks/system1_tutorial.ipynb
```

The supplied notebook was checked in a fresh Python kernel with live calls disabled, including the optional fixture-based LangGraph workflow. The deck was rendered at desktop and phone sizes. Live service calls were not used to create the retained tutorial results. [Verification notes](docs/verification.md) describe the checks and their limits.
