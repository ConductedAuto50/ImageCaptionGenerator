import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, Tuple

import requests

# Paths
DATASET_PATH = Path("/home/saksham/NLP/datasets/instagram_data/captions_csv_cleaned.csv")
OUTPUT_PATH = DATASET_PATH.with_name(f"{DATASET_PATH.stem}_with_emotions{DATASET_PATH.suffix}")

# Model and API settings
MODEL_NAME = "gemma3:4b"
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
ALLOWED_EMOTIONS = ["casual", "formal", "poetic", "funny"]


PROMPT_TEMPLATE = """Classify the tone of this Instagram caption.

TASK: Identify the TWO most fitting tones from: casual, formal, poetic, funny.

TONE DEFINITIONS:
- casual: Relaxed, conversational, uses slang/abbreviations (hey, lol, gonna, vibes, etc.)
- formal: Professional, polished, proper grammar, announcements, achievements
- poetic: Lyrical, metaphorical, emotional depth, imagery, philosophical
- funny: Humorous, jokes, sarcasm, playful, witty, self-deprecating

RULES:
1. primary = the SINGLE best-fitting tone
2. secondary = the SECOND best tone (must differ from primary)
3. Use only lowercase labels

Caption: {caption}

{{"primary":"","secondary":""}}"""


def build_payload(caption: str) -> Dict:
    """Construct the payload for the Ollama generate endpoint."""
    prompt = PROMPT_TEMPLATE.format(caption=caption.strip())
    return {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "format": {
            "type": "object",
            "properties": {
                "primary": {"type": "string", "enum": ALLOWED_EMOTIONS},
                "secondary": {"type": "string", "enum": ALLOWED_EMOTIONS},
            },
            "required": ["primary", "secondary"],
            "additionalProperties": False,
        },
        "options": {
            "temperature": 0.1,
            "num_predict": 32,  # Limit output tokens for speed
        },
    }


def normalize_label(value: str) -> str:
    """Normalize a label and ensure it is one of the allowed values."""
    if not isinstance(value, str):
        return ""
    label = value.strip().lower()
    return label if label in ALLOWED_EMOTIONS else ""


def fallback_labels(caption: str) -> Tuple[str, str, list[str]]:
    """Lightweight heuristic fallback when the model output is unusable.
    Returns (primary, secondary, full_ranking) based on keyword heuristics."""
    text = caption.lower()
    
    # Score each emotion based on keyword matches
    scores = {emo: 0 for emo in ALLOWED_EMOTIONS}
    
    # Funny indicators
    if any(word in text for word in ["haha", "lol", "funny", "joke", "😂", "🤣", "lmao", "rofl", "hilarious"]):
        scores["funny"] += 3
    
    # Poetic indicators
    if any(word in text for word in ["poem", "poetic", "dream", "moon", "stars", "rain", "✨", "🌙", "soul", "heart", "eternal", "whisper", "silence"]):
        scores["poetic"] += 3
    
    # Casual indicators
    if any(word in text for word in ["hey", "yo", "gonna", "wanna", "love", "fun", "cool", "thanks", "guys", "omg", "btw", "ngl"]):
        scores["casual"] += 3
    
    # Formal indicators
    if any(word in text for word in ["announce", "pleased", "professional", "official", "honored", "opportunity", "grateful", "achievement"]):
        scores["formal"] += 3
    
    # Default boost to casual if nothing matches strongly
    if all(s == 0 for s in scores.values()):
        scores["casual"] = 1
    
    # Sort emotions by score (descending), then alphabetically for ties
    ranked = sorted(ALLOWED_EMOTIONS, key=lambda e: (-scores[e], e))
    
    primary = ranked[0]
    secondary = ranked[1]
    
    return primary, secondary, ranked


def derive_full_ranking(primary: str, secondary: str) -> list[str]:
    """Derive full 4-emotion ranking from primary and secondary."""
    ranking = [primary, secondary]
    # Add remaining emotions in default order
    for emo in ALLOWED_EMOTIONS:
        if emo not in ranking:
            ranking.append(emo)
    return ranking


def classify_caption(session: requests.Session, caption: str) -> Tuple[str, str, list[str]]:
    """Call Ollama to classify a caption into primary/secondary emotions."""
    if not caption or not caption.strip():
        primary, secondary, ranking = fallback_labels("")
        return primary, secondary, ranking

    payload = build_payload(caption)
    try:
        response = session.post(OLLAMA_URL, json=payload, timeout=60)
        response.raise_for_status()
        body = response.json()
        raw = body.get("response", "").strip()
        parsed = json.loads(raw)
        primary = normalize_label(parsed.get("primary", ""))
        secondary = normalize_label(parsed.get("secondary", ""))
    except Exception:
        primary, secondary = "", ""

    # If primary is invalid, use fallback
    if not primary or primary not in ALLOWED_EMOTIONS:
        primary, secondary, ranking = fallback_labels(caption)
        return primary, secondary, ranking

    # If secondary is invalid or same as primary, pick next available
    if not secondary or secondary not in ALLOWED_EMOTIONS or secondary == primary:
        secondary = next((emo for emo in ALLOWED_EMOTIONS if emo != primary), ALLOWED_EMOTIONS[0])

    # Derive full ranking from primary and secondary
    ranking = derive_full_ranking(primary, secondary)

    return primary, secondary, ranking


def add_emotion_columns(limit: int = 0, progress_interval: int = 200) -> None:
    """Read the CSV, add emotion columns, and write to a new file."""
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")

    # Avoid clobbering a previous run's output.
    if OUTPUT_PATH.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {OUTPUT_PATH}")

    with DATASET_PATH.open("r", newline="", encoding="utf-8") as infile:
        reader = csv.reader(infile)
        try:
            header = next(reader)
        except StopIteration:
            raise ValueError("CSV appears to be empty.")

        # Avoid duplicating columns if they already exist.
        already_has = (
            len(header) >= 5
            and header[3].strip().lower() == "primary emotion"
            and header[4].strip().lower() == "secondary emotion"
        )

        if already_has:
            raise ValueError("File already contains emotion columns; aborting to avoid duplication.")

        new_header = (
            header[:3]
            + ["Primary Emotion", "Secondary Emotion", "Rank 1 Emotion", "Rank 2 Emotion", "Rank 3 Emotion", "Rank 4 Emotion"]
            + header[3:]
        )

        with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as outfile, requests.Session() as session:
            writer = csv.writer(outfile)
            writer.writerow(new_header)

            for idx, row in enumerate(reader, start=1):
                caption = row[2] if len(row) > 2 else ""
                primary, secondary, ranking = classify_caption(session, caption)
                new_row = row[:3] + [primary, secondary] + ranking + row[3:]
                writer.writerow(new_row)

                # Print per-row classification for visibility
                print(f"Row {idx}: primary={primary}, secondary={secondary}, ranking={ranking}")

                if progress_interval and idx % progress_interval == 0:
                    print(f"Processed {idx} rows...")

                if limit and idx >= limit:
                    print(f"Limit of {limit} rows reached; stopping early.")
                    break

    print(f"New file written to {OUTPUT_PATH}")
    print(f"Original file left unchanged at {DATASET_PATH}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add ranked emotion labels (primary, secondary) to captions CSV using Ollama gemma3:4b."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process only the first N rows (useful for quick verification). Default: 0 (process all).",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=200,
        help="Print a progress message every N rows. Set to 0 to disable.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = time.time()
    add_emotion_columns(limit=args.limit, progress_interval=args.progress_interval)
    elapsed = time.time() - start
    print(f"Completed in {elapsed:.2f}s")


if __name__ == "__main__":
    main()

