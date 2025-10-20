#!/usr/bin/env python3
"""
House Diagnosis Prompt Generator and LLM Responder using Gemini API.

Mimics the OpenAI code behavior exactly:
1) Collect structured prompts (Prompt, Expected Response, Disease)
2) Call Gemini to get actual LLM responses
3) Compute fuzzy accuracy
4) Save Excel files at each stage
"""

import os
import sys
import json
import time
import logging
import re
from pathlib import Path
from typing import List, Optional
from difflib import SequenceMatcher

import pandas as pd
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from pydantic import BaseModel
import google.generativeai as genai

from constants import SYSTEM_PROMPT, HOUSE_EPISODE_TITLES, BASE_URL

# ---------- Config ----------
GENERATED_DATA_DIR = Path("generated_data")
GENERATED_DATA_DIR.mkdir(parents=True, exist_ok=True)

CALL_DELAY_SECONDS = 0.5
MAX_API_RETRIES = 3
RETRY_BACKOFF_FACTOR = 2.0
MAX_UNKNOWN_RETRIES = 2
MODEL_NAME = "gemini-2.5-pro"

# ---------- Logging ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------- Load env ----------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.error("Missing GEMINI_API_KEY in environment (.env).")
    sys.exit(1)
genai.configure(api_key=GEMINI_API_KEY)
_model = genai.GenerativeModel(MODEL_NAME)

# ---------- Models ----------
class EpisodeInformation(BaseModel):
    prompt: str
    expected_llm_response: str
    disease_name: str

# ---------- Helpers ----------
def _extract_json_from_text(text: str) -> Optional[dict]:
    try:
        return json.loads(text)
    except Exception:
        pass
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start:i+1]
                try:
                    return json.loads(candidate)
                except Exception:
                    return None
    return None

def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(text).lower())

def is_close_match(expected: str, actual: str, threshold: float = 0.8) -> bool:
    expected_norm = normalize(expected)
    actual_norm = normalize(actual)
    if expected_norm and expected_norm in actual_norm:
        return True
    for word in str(actual).split():
        word_norm = normalize(word)
        if SequenceMatcher(None, expected_norm, word_norm).ratio() >= threshold:
            return True
    return False

# ---------- Episode Parsing ----------
class ParsedEpisode:
    def __init__(self, url: str, episode_name: str):
        self.url = url
        self.episode_name = episode_name
        self.content = self._fetch_content()

    def _fetch_content(self) -> str:
        try:
            resp = requests.get(self.url, timeout=20)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            return soup.get_text(separator="\n")
        except Exception as e:
            logger.error("Failed to fetch %s: %s", self.url, e)
            return ""

    def build_llm_prompt(self) -> str:
        return (
            f"{SYSTEM_PROMPT}\n\n"
            "Create a structured JSON with the following fields: prompt, expected_llm_response, disease_name.\n"
            f"### EPISODE INFO ###\n{self.content}"
        )

# ---------- Gemini API call ----------
def call_gemini_with_retries(prompt_text: str, max_retries: int = MAX_API_RETRIES) -> EpisodeInformation:
    attempt = 0
    delay = CALL_DELAY_SECONDS
    last_exception = None

    while attempt <= max_retries:
        try:
            resp = _model.generate_content(
                contents=[{"role": "user", "parts": [prompt_text]}],
                generation_config=genai.types.GenerationConfig(
                    temperature=0,
                    max_output_tokens=15000
                )
            )
            # extract text
            text = getattr(resp, "text", None)
            if not text and hasattr(resp, "candidates"):
                text = resp.candidates[0].content.parts[0].text
            text = (text or "").strip()

            parsed = _extract_json_from_text(text)
            if parsed:
                return EpisodeInformation(
                    prompt=parsed.get("prompt", "").strip() or prompt_text,
                    expected_llm_response=parsed.get("expected_llm_response", "").strip() or text,
                    disease_name=parsed.get("disease_name", "").strip() or "Unknown"
                )
            else:
                return EpisodeInformation(prompt=prompt_text, expected_llm_response=text, disease_name="Unknown")

        except Exception as e:
            last_exception = e
            attempt += 1
            logger.warning("Gemini call failed (attempt %d/%d): %s", attempt, max_retries, e)
            time.sleep(delay)
            delay *= RETRY_BACKOFF_FACTOR

    raise last_exception

# ---------- Step 1: Collect structured prompts ----------
def collect_episode_prompts() -> pd.DataFrame:
    records = []
    for ep_name in HOUSE_EPISODE_TITLES:
        url = f"{BASE_URL}{ep_name.replace(' ', '_')}"
        logger.info("Processing episode: %s", ep_name)
        episode = ParsedEpisode(url, ep_name)
        raw_prompt = episode.build_llm_prompt()

        try:
            info = call_gemini_with_retries(raw_prompt)
            records.append({
                "Episode Name": ep_name,
                "Prompt": info.prompt,
                "Expected LLM Response": info.expected_llm_response,
                "Disease": info.disease_name
            })
        except Exception as e:
            logger.error("Gemini failed for %s: %s", ep_name, e)
            records.append({
                "Episode Name": ep_name,
                "Prompt": raw_prompt,
                "Expected LLM Response": f"ERROR: {e}",
                "Disease": "Unknown"
            })
        time.sleep(CALL_DELAY_SECONDS)
    df = pd.DataFrame(records)
    df.to_excel(GENERATED_DATA_DIR / "House_Diagnosis_structured_geminipro.xlsx", index=False)
    return df

# ---------- Step 2: Generate actual LLM responses ----------
def generate_actual_responses(df: pd.DataFrame) -> pd.DataFrame:
    responses = []
    models = []

    for idx, row in df.iterrows():
        prompt = row["Prompt"]
        logger.info("Generating actual response for row %d", idx)

        for attempt in range(MAX_API_RETRIES):
            try:
                resp = _model.generate_content(
                    contents=[{"role": "user", "parts": [prompt]}],
                    generation_config=genai.types.GenerationConfig(
                        temperature=0,
                        max_output_tokens=15000
                    )
                )
                text = getattr(resp, "text", None)
                if not text and hasattr(resp, "candidates"):
                    text = resp.candidates[0].content.parts[0].text
                text = (text or "").strip()
                responses.append(text)
                models.append(MODEL_NAME)
                break  # success, break retry loop

            except Exception as e:
                err_str = str(e)
                if "429" in err_str:
                    # Extract suggested retry time (if available)
                    match = re.search(r"retry in ([0-9.]+)s", err_str)
                    if match:
                        retry_secs = float(match.group(1))
                    else:
                        retry_secs = 60  # default fallback
                    logger.warning("Quota exceeded. Sleeping for %.1f seconds...", retry_secs)
                    time.sleep(retry_secs)
                    continue
                else:
                    logger.error("Error at row %d: %s", idx, e)
                    responses.append(f"ERROR: {e}")
                    models.append(None)
                    break  # don't retry for non-429 errors

        time.sleep(CALL_DELAY_SECONDS)

    df["actual_llm_response"] = responses
    df["model_used"] = models
    df.to_excel(GENERATED_DATA_DIR / "House_Diagnosis_structured_geminipro_with_responses.xlsx", index=False)
    return df


# ---------- Step 3: Compute accuracy ----------
def compute_accuracy(df: pd.DataFrame) -> float:
    correct_flags = []
    for _, row in df.iterrows():
        disease = str(row.get("Disease", ""))
        response = str(row.get("actual_llm_response", ""))
        correct_flags.append(is_close_match(disease, response))
    df["is_correct"] = correct_flags
    df.to_excel(GENERATED_DATA_DIR / "House_Diagnosis_with_accuracy_geminipro.xlsx", index=False)
    accuracy = sum(correct_flags) / len(correct_flags) if correct_flags else 0
    logger.info("Accuracy: %.2f%% (%d/%d correct)", accuracy * 100, sum(correct_flags), len(correct_flags))
    return accuracy

# ---------- Main for reusing structured Excel ----------
def main_from_existing_excel():
    # Load the already structured file
    input_path = GENERATED_DATA_DIR / "House_Diagnosis_structured_geminipro.xlsx"
    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        sys.exit(1)

    logger.info("Loading existing structured data from %s", input_path)
    df = pd.read_excel(input_path)

    # Step 2: Generate actual responses
    df_with_responses = generate_actual_responses(df)
    df_with_responses.to_excel(GENERATED_DATA_DIR / "House_Diagnosis_structured_geminipro_with_responses.xlsx", index=False)

    # Step 3: Compute accuracy
    accuracy = compute_accuracy(df_with_responses)
    logger.info("Final Accuracy: %.2f%%", accuracy * 100)

def accuracy_by_season(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-season accuracy using the 'Is Correct' column,
    based on known row index ranges for each season.

    Args:
        df: DataFrame containing 'Is Correct' and episode rows in show order.

    Returns:
        DataFrame summarizing accuracy per season.
    """

    # Define mapping of row index ranges (1-based, so adjust for 0-based pandas index)
    season_ranges = {
        1: (1, 23),
        2: (24, 47),
        3: (48, 71),
        4: (72, 87),
        5: (88, 111),
        6: (112, 132),
        7: (133, 155),
        8: (156, 177),
    }

    # Create a new 'Season' column based on row number
    df = df.copy()
    df["Season"] = None

    for season, (start, end) in season_ranges.items():
        mask = ((df.index + 1) >= start) & ((df.index + 1) <= end)
        df.loc[mask, "Season"] = f"Season {season}"

    # Compute accuracy per season
    season_stats = (
        df.groupby("Season")["is_correct"]
        .agg(["count", "sum"])
        .rename(columns={"count": "Total Episodes", "sum": "Correct Predictions"})
        .reset_index()
    )
    season_stats["Accuracy (%)"] = (
        season_stats["Correct Predictions"] / season_stats["Total Episodes"] * 100
    ).round(2)

    # Log and save
    logger.info("\nPer-season accuracy:\n%s", season_stats.to_string(index=False))
    #output_path = GENERATED_DATA_DIR / "House_Diagnosis_accuracy_by_season.xlsx"
    #season_stats.to_excel(output_path, index=False, engine="openpyxl")
    #logger.info("Saved per-season accuracy to %s", output_path)

    return season_stats


# ---------- Main ----------
def main():
    df = collect_episode_prompts()
    generate_actual_responses(df)
    compute_accuracy(df)
    accuracy_by_season(df)

if __name__ == "__main__":
    main()
    
