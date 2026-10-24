"""Optional explicit Chromium executable override for artifact review."""
import os
from pathlib import Path


def launch_browser(playwright):
    executable = os.getenv("MM_CHROMIUM_EXECUTABLE")
    if executable:
        if not Path(executable).is_file():
            raise ValueError("MM_CHROMIUM_EXECUTABLE must name an installed browser executable")
        return playwright.chromium.launch(executable_path=executable)
    return playwright.chromium.launch()
