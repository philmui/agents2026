"""Execute every lesson in a fresh kernel, saving outputs only after success.

Run from the module: uv run python tools/execute_notebooks.py
The validation run deliberately disables model downloads and API calls.
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import nbformat
from jupyter_client import KernelManager
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    os.environ["MM_RUN_MODELS"] = "0"
    os.environ["MM_RUN_LIVE"] = "0"
    results = []
    paths = sorted(ROOT.glob("[0-9][0-9]_*.ipynb"))
    if not paths:
        raise SystemExit("No notebooks found; build the lessons first.")
    for path in paths:
        start = time.perf_counter()
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        manager = KernelManager(kernel_name="python3")
        # Use this uv environment even if the user's global kernels differ.
        manager.kernel_spec.argv[0] = sys.executable
        client = NotebookClient(
            notebook,
            km=manager,
            timeout=180,
            allow_errors=False,
            resources={"metadata": {"path": str(ROOT)}},
        )
        try:
            client.execute()
        finally:
            if manager.has_kernel:
                manager.shutdown_kernel(now=True)
        nbformat.write(notebook, path)
        result = {
            "notebook": path.name,
            "cells": len(notebook.cells),
            "code_cells": sum(c.cell_type == "code" for c in notebook.cells),
            "seconds": round(time.perf_counter() - start, 2),
            "status": "passed",
        }
        results.append(result)
        print(json.dumps(result), flush=True)
    report = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "mode": "local; MM_RUN_MODELS=0; MM_RUN_LIVE=0",
        "results": results,
    }
    output = ROOT / "artifacts" / "notebook_execution.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
