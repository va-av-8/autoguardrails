#!/usr/bin/env python3
"""Пересобирает §3 base64-ячейку в kaggle_automl_baselines_jailbreak.ipynb."""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

KEYS = [
    "tasks/__init__.py",
    "tasks/jailbreak_detection/__init__.py",
    "tasks/jailbreak_detection/src/__init__.py",
    "tasks/jailbreak_detection/src/metrics.py",
    "tasks/jailbreak_detection/scripts/prepare_data.py",
    "tasks/jailbreak_detection/scripts/run_automl_baselines.py",
]


def build_payload(repo_root: Path) -> dict[str, str]:
    return {
        rel: (repo_root / rel).read_text(encoding="utf-8") if (repo_root / rel).is_file() else ""
        for rel in KEYS
    }


def cell_source(blob_lines: str) -> str:
    return (
        "import base64\n"
        "import json\n"
        "from pathlib import Path\n"
        "\n"
        "_BLOB = \"\"\"\n"
        f"{blob_lines}\n"
        "\"\"\"\n"
        "\n"
        "_P = json.loads(base64.b64decode(_BLOB.encode(\"ascii\")).decode(\"utf-8\"))\n"
        "for rel, txt in _P.items():\n"
        "    p = PROJECT_ROOT / rel\n"
        "    p.parent.mkdir(parents=True, exist_ok=True)\n"
        "    p.write_text(txt, encoding=\"utf-8\")\n"
        "print(\"Written\", len(_P), \"files under\", PROJECT_ROOT)\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[3])
    ap.add_argument(
        "--notebook",
        type=Path,
        default=Path(__file__).resolve().parent / "kaggle_automl_baselines_jailbreak.ipynb",
    )
    args = ap.parse_args()

    raw = json.dumps(build_payload(args.repo_root), ensure_ascii=False)
    wrapped = "\n".join(
        base64.b64encode(raw.encode("utf-8")).decode("ascii")[i : i + 100]
        for i in range(0, len(base64.b64encode(raw.encode("utf-8")).decode("ascii")), 100)
    )
    src = cell_source(wrapped)
    compile(src, "<cell>", "exec")

    nb = json.loads(args.notebook.read_text(encoding="utf-8"))
    for i, c in enumerate(nb["cells"]):
        if c.get("cell_type") == "code" and "_BLOB" in "".join(c.get("source", [])):
            nb["cells"][i]["source"] = [line + "\n" for line in src.splitlines()]
            if not nb["cells"][i]["source"][-1].endswith("\n"):
                nb["cells"][i]["source"][-1] += "\n"
            args.notebook.write_text(json.dumps(nb, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print("Updated cell", i, "payload_chars", len(raw))
            return
    raise SystemExit("base64 cell not found — run notebook generator first")


if __name__ == "__main__":
    main()
