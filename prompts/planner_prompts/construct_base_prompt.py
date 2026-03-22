"""
Inject optimization memory into planner base prompt.
From AccelOpt construct_base_prompt.py, adapted for our structure.
"""
import json
import argparse
from pathlib import Path


def inject_memory(base_prompt: str, memory_entries: list) -> str:
    """
    Append optimization memory entries to planner base prompt.
    Each entry has {"title": "...", "summary": "..."}.
    """
    additional = ""
    for item in memory_entries:
        title = item.get("title", "")
        if title == "**No optimization found**":
            continue
        summary = item.get("summary", "")
        if summary:
            additional += f"{summary}\n\n"

    if additional:
        return base_prompt + "\n\n# Optimization Experiences\n" + additional
    return base_prompt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-base-prompt", required=True)
    parser.add_argument("--memory-file", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    base_prompt = Path(args.original_base_prompt).read_text()

    memory_path = Path(args.memory_file)
    if memory_path.exists():
        memory_entries = json.loads(memory_path.read_text())
    else:
        memory_entries = []

    result = inject_memory(base_prompt, memory_entries)
    Path(args.output).write_text(result)
    print(f"Wrote {len(result)} chars to {args.output} ({len(memory_entries)} experiences)")


if __name__ == "__main__":
    main()
