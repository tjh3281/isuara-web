"""
Print the notebook cells that build and export the model.

The web build needs the architecture and the weights; the notebook is the only
record of both. Run with a substring to filter cells.

    python tools/read_notebook.py inner_weights
"""

import json
import sys
from pathlib import Path

NOTEBOOK = Path(__file__).resolve().parents[3] / "iSuara_Train_V3_1_5_WithOutput.ipynb"


def main() -> None:
    needle = sys.argv[1] if len(sys.argv) > 1 else "inner_weights"
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))

    for i, cell in enumerate(nb["cells"]):
        source = "".join(cell["source"])
        if needle.lower() in source.lower():
            print(f"\n{'=' * 70}\nCELL {i} ({cell['cell_type']})\n{'=' * 70}")
            print(source)


if __name__ == "__main__":
    main()
