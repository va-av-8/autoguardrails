#!/usr/bin/env python3
"""Пересобирает ячейку §3 ноутбука kaggle_heavy_presets_jailbreak.ipynb (base64-payload файлов tasks/)."""

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
    "tasks/jailbreak_detection/scripts/run_autointent.py",
]


def build_payload(repo_root: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for rel in KEYS:
        fp = repo_root / rel
        out[rel] = fp.read_text(encoding="utf-8") if fp.is_file() else ""
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[3],
        help="Корень репозитория jailbreak_detection_analysis",
    )
    ap.add_argument(
        "--notebook",
        type=Path,
        default=Path(__file__).resolve().parent / "kaggle_heavy_presets_jailbreak.ipynb",
    )
    args = ap.parse_args()

    raw = json.dumps(build_payload(args.repo_root), ensure_ascii=False)
    blob = base64.b64encode(raw.encode("utf-8")).decode("ascii")
    wrapped = "\n".join(blob[i : i + 100] for i in range(0, len(blob), 100))

    cell_src = (
        "import base64\n"
        "import json\n"
        "from pathlib import Path\n"
        "\n"
        "# Файлы проекта в base64: иначе при разбиении ячейки на куски ломается литерал json.loads('…').\n"
        "_BLOB = \"\"\"\n"
        f"{wrapped}\n"
        "\"\"\"\n"
        "\n"
        "_P = json.loads(base64.b64decode(_BLOB.encode(\"ascii\")).decode(\"utf-8\"))\n"
        "\n"
        "for rel, txt in _P.items():\n"
        "    p = PROJECT_ROOT / rel\n"
        "    p.parent.mkdir(parents=True, exist_ok=True)\n"
        "    p.write_text(txt, encoding=\"utf-8\")\n"
        "print(\"Written\", len(_P), \"files\")\n"
    )
    compile(cell_src, "<cell>", "exec")

    nb = json.loads(args.notebook.read_text(encoding="utf-8"))
    for i, c in enumerate(nb["cells"]):
        if c.get("cell_type") != "code":
            continue
        src = "".join(c.get("source", []))
        if "_BLOB" in src and "PROJECT_ROOT" in src and "Written" in src and len(src) > 5000:
            nb["cells"][i]["source"] = [cell_src]
            args.notebook.write_text(json.dumps(nb, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print("Updated notebook cell index", i, "json_chars", len(raw), "b64_chars", len(blob))
            return
    raise SystemExit("target cell not found")


if __name__ == "__main__":
    main()
