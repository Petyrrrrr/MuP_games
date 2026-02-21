#!/usr/bin/env python3
import json
from pathlib import Path

kernelspec = {
    "display_name": "Python (venv)",
    "language": "python",
    "name": "venv"
}

notebooks = list(Path(".").glob("*.ipynb"))
for nb_path in notebooks:
    with open(nb_path) as f:
        nb = json.load(f)
    nb["metadata"]["kernelspec"] = kernelspec
    with open(nb_path, "w") as f:
        json.dump(nb, f, indent=1)
    print(f"Updated: {nb_path}")

print(f"\nDone — {len(notebooks)} notebooks updated.")
