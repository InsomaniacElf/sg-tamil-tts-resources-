from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import yaml

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
README_FILE = ROOT / "README.md"
TAMIL_FILE = ROOT / "tamil-tts.md"
CONFIG_FILE = Path(__file__).resolve().with_name("config.yaml")


def load_json(path: Path) -> list[dict]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8") or "[]")
    return data if isinstance(data, list) else []


def deduplicate(entries: Iterable[dict]) -> list[dict]:
    deduped: list[dict] = []
    seen: set[str] = set()
    for entry in entries:
        key = (entry.get("url") or entry.get("title") or "").strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(entry)
    return deduped


def render_table(entries: list[dict]) -> str:
    if not entries:
        return "| Title | Type | URL | Notes |\n| --- | --- | --- | --- |\n| _No entries yet_ | - | - | - |"

    rows = ["| Title | Type | URL | Notes |", "| --- | --- | --- | --- |"]
    for item in entries:
        title = item.get("title", "-").replace("|", "\\|")
        category = item.get("type", "-").replace("|", "\\|")
        url = item.get("url", "-")
        notes = item.get("notes", "-").replace("|", "\\|")
        rows.append(f"| {title} | {category} | {url} | {notes} |")
    return "\n".join(rows)


def render_doc(title: str, description: str, entries: list[dict], keywords: list[str]) -> str:
    keyword_text = ", ".join(keywords)
    return "\n".join(
        [
            f"# {title}",
            "",
            description,
            "",
            f"_Keywords:_ {keyword_text}",
            "",
            render_table(entries),
            "",
        ]
    )


def main() -> None:
    config = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8")) or {}
    tamil_entries = deduplicate(load_json(DATA_DIR / "entries.json"))
    general_entries = deduplicate(load_json(DATA_DIR / "entries_tts.json"))

    (DATA_DIR / "entries.json").write_text(
        json.dumps(tamil_entries, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (DATA_DIR / "entries_tts.json").write_text(
        json.dumps(general_entries, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    README_FILE.write_text(
        render_doc(
            "sg-tamil-tts-resources-",
            "A curated, searchable list of datasets, models, research papers, and tools for building Text-to-Speech systems.",
            general_entries,
            config.get("general_keywords", []),
        ),
        encoding="utf-8",
    )

    TAMIL_FILE.write_text(
        render_doc(
            "Singapore Tamil TTS Resources",
            "Auto-generated Tamil-focused subset of the resource list.",
            tamil_entries,
            config.get("tamil_keywords", []),
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
