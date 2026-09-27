"""Create AI-written CVs with a local LLM via Ollama (free, runs on your computer).

Two kinds, because both happen in real life:
  scratch: the AI writes a whole CV from a short description
  polish:  the AI rewrites a real human CV to "make it more professional"

Examples:
    ollama pull llama3.2
    python generate.py --model llama3.2 --n 100
    python generate.py --model qwen2.5:3b --n 100     # a second generator

The script is resumable: stop it any time with Ctrl+C and run the same command again.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

from detector.data import load_human_csv

ROOT = Path(__file__).resolve().parent
DEFAULT_HUMAN_CSV = ROOT / "data" / "human" / "Resume.csv"
DEFAULT_OUT = ROOT / "data" / "generated"

SENIORITY = ["entry-level", "mid-level", "senior"]
YEARS = {"entry-level": "0-2", "mid-level": "3-7", "senior": "8-15"}

SCRATCH_PROMPT = """Write a complete, realistic resume for a {seniority} professional in the field of {category} with {years} years of experience.
Include a summary, work experience with bullet points, education and skills.
Use the exact text "Company Name" for every employer and "City, State" for every location. Do not include a person's name or contact details.
Output only the resume text, with no introduction or closing remarks."""

POLISH_PROMPT = """Improve the following resume so that it sounds more professional and compelling to recruiters.
Keep the same jobs, dates and facts. Keep "Company Name" and "City, State" exactly as written.
Output only the improved resume text, with no introduction or closing remarks.

RESUME:
{resume}"""

PREAMBLE_RE = re.compile(r"^(here('s| is)|sure|certainly|below is)\b.*$", re.IGNORECASE | re.MULTILINE)
CLOSING_RE = re.compile(r"\n\s*(note\b|i hope\b|feel free\b|let me know\b|this (revised )?resume\b).*$", re.IGNORECASE | re.DOTALL)


def clean_generation(text: str) -> str:
    """Remove chatty lines like "Here is your resume:" that would give the AI away trivially."""
    text = CLOSING_RE.sub("", text)
    text = PREAMBLE_RE.sub("", text, count=1)
    return text.strip()


def ollama_generate(prompt: str, model: str, host: str, temperature: float, seed: int) -> str:
    body = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature, "seed": seed, "num_predict": 1200},
        }
    ).encode()
    request = urllib.request.Request(f"{host}/api/generate", data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            return json.loads(response.read())["response"]
    except urllib.error.HTTPError as err:
        detail = err.read().decode(errors="ignore")
        sys.exit(f"Ollama returned an error: {detail}\nDid you run `ollama pull {model}`?")
    except urllib.error.URLError:
        sys.exit("Can't reach Ollama. Open the Ollama app (or run `ollama serve`) and try again.")


def existing_count(path: Path, mode: str) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as f:
        return sum(json.loads(line)["mode"] == mode for line in f if line.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Ollama model name, e.g. llama3.2")
    parser.add_argument("--n", type=int, default=100, help="CVs per mode")
    parser.add_argument("--modes", nargs="+", choices=["scratch", "polish"], default=["scratch", "polish"])
    parser.add_argument("--human-csv", type=Path, default=DEFAULT_HUMAN_CSV)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--host", default="http://localhost:11434")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    humans = load_human_csv(args.human_csv)
    categories = sorted(humans["category"].unique())
    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / f"{re.sub(r'[^a-zA-Z0-9]+', '_', args.model)}.jsonl"
    rng = random.Random(f"{args.seed}-{args.model}")

    for mode in args.modes:
        done = existing_count(out_path, mode)
        # Polish sources are drawn in a fixed order so a resumed run continues where it stopped.
        sources = humans.sample(frac=1, random_state=args.seed).reset_index(drop=True)
        for i in range(done, args.n):
            if mode == "scratch":
                seniority = rng.choice(SENIORITY)
                category = rng.choice(categories)
                prompt = SCRATCH_PROMPT.format(
                    seniority=seniority, category=category.replace("-", " ").lower(), years=YEARS[seniority]
                )
                source_id = ""
            else:
                source = sources.iloc[i % len(sources)]
                resume = " ".join(source["text"].split()[:500])
                prompt, category, source_id = POLISH_PROMPT.format(resume=resume), source["category"], source["id"]

            text = clean_generation(
                ollama_generate(prompt, args.model, args.host, temperature=rng.uniform(0.6, 1.0), seed=args.seed + i)
            )
            row = {
                "id": f"{args.model}-{mode}-{i}",
                "text": text,
                "generator": args.model,
                "mode": mode,
                "category": category,
                "source_id": source_id,
            }
            with out_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            print(f"[{mode} {i + 1}/{args.n}] {len(text.split())} words")

    print(f"Done. Saved to {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
