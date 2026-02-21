#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

LOCAL_KERNELSPEC = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}
REMOTE_KERNELSPEC = {
    "display_name": "Python (venv)",
    "language": "python",
    "name": "venv",
}

def main():
    parser = argparse.ArgumentParser(description="Set Jupyter kernel for all .ipynb notebooks.")
    parser.add_argument("--remote", type=int, choices=[0, 1], required=True,
                        help="0 = local Python interpreter (Python 3); 1 = venv interpreter")
    args = parser.parse_args()
    kernelspec = REMOTE_KERNELSPEC if args.remote else LOCAL_KERNELSPEC

    notebooks = list(Path(".").glob("*.ipynb"))
    for nb_path in notebooks:
        with open(nb_path) as f:
            nb = json.load(f)
        nb["metadata"]["kernelspec"] = kernelspec
        with open(nb_path, "w") as f:
            json.dump(nb, f, indent=1)
        print(f"Updated: {nb_path}")

    print(f"\nDone — {len(notebooks)} notebooks updated.")

if __name__ == "__main__":
    main()
