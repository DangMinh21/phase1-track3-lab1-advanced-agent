from __future__ import annotations

import json
import random
from pathlib import Path

import typer

app = typer.Typer(add_completion=False)


def _to_context(raw_context: list) -> list[dict]:
    mapped: list[dict] = []
    for item in raw_context:
        if not isinstance(item, list) or len(item) != 2:
            continue
        title, sentences = item
        if isinstance(sentences, list):
            text = " ".join(s.strip() for s in sentences if isinstance(s, str))
        else:
            text = str(sentences)
        mapped.append({"title": str(title), "text": text})
    return mapped


@app.command()
def main(
    input_path: str = typer.Option(..., help="Path to raw HotpotQA json"),
    output_path: str = "data/hotpot_100.json",
    num_samples: int = 120,
    seed: int = 42,
) -> None:
    raw = json.loads(Path(input_path).read_text(encoding="utf-8"))

    rng = random.Random(seed)
    rng.shuffle(raw)

    rows: list[dict] = []
    for item in raw:
        qid = item.get("_id")
        question = item.get("question")
        answer = item.get("answer")
        context = _to_context(item.get("context", []))
        level = str(item.get("level", "medium")).lower()
        if level not in {"easy", "medium", "hard"}:
            level = "medium"

        if not qid or not question or answer is None or not context:
            continue

        rows.append(
            {
                "qid": str(qid),
                "difficulty": level,
                "question": str(question),
                "gold_answer": str(answer),
                "context": context,
            }
        )
        if len(rows) >= num_samples:
            break

    if len(rows) < 100:
        raise ValueError(f"Only {len(rows)} valid rows collected. Need at least 100.")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {len(rows)} rows -> {out}")


if __name__ == "__main__":
    app()