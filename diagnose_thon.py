#!/usr/bin/env python3
"""
House Diagnosis Prompt Generator and LLM Responder.

This script scrapes medical case details from House episode pages,
creates structured prompts for an LLM, invokes the LLM to produce
expected diagnoses, and saves results to Excel files.
"""

import os
import sys
import logging
from pathlib import Path
from typing import List

import pandas as pd
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
from difflib import SequenceMatcher


from constants import SYSTEM_PROMPT, HOUSE_EPISODE_TITLES, BASE_URL
GENERATED_DATA_DIR = Path("generated_data")
GENERATED_DATA_DIR.mkdir(parents=True, exist_ok=True)


# Configure logger with a clear format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.error("Missing OpenAI API key. Add OPENAI_API_KEY to your .env file.")
    sys.exit(1)

# Initialize the OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

class EpisodeInformation(BaseModel):
    """Structured response from LLM for a single episode."""
    prompt: str
    expected_llm_response: str
    disease_name: str

class ParsedEpisode:
    """Fetch and parse episode web page to extract raw text content."""

    def __init__(self, url: str, episode_name: str):
        self.url = url
        self.episode_name = episode_name
        self.content = self._fetch_content()

    def _fetch_content(self) -> str:
        """Retrieve and parse HTML content from the URL."""
        try:
            response = requests.get(self.url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            return soup.get_text(separator="\n")
        except requests.RequestException as e:
            logger.error("Failed to fetch %s: %s", self.url, e)
            return ""

    def build_llm_prompt(self) -> str:
        """Compose the prompt text to send to the LLM."""
        return (
            "Use details in INFORMATION for creating your LLM prompt, "
            "required medical answer, and the disease name.\n\n"
            f"### INFORMATION ###\n{self.content}\n\n"
            "Based on this information, generate a detailed prompt for a "
            "medical professional track LLM to diagnose the patient. Also, "
            "provide the expected medical answer that the LLM should give, "
            "and explicitly state the exact disease name."
        )

def collect_episode_prompts() -> pd.DataFrame:
    """
    Scrape episode pages and generate structured prompts and expected responses.

    Returns:
        DataFrame with columns: Episode Name, Prompt, Expected LLM Response, Disease
    """
    records: List[dict] = []

    for ep_name in HOUSE_EPISODE_TITLES:
        url = f"{BASE_URL}{ep_name.replace(' ', '_')}"
        logger.info("Processing episode: %s", ep_name)
        episode = ParsedEpisode(url, ep_name)

        raw_prompt = episode.build_llm_prompt()
        try:
            # Parse structured response into EpisodeInformation model
            resp = client.beta.chat.completions.parse(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": raw_prompt},
                ],
                response_format=EpisodeInformation,
            )
            info = resp.choices[0].message.parsed
            records.append({
                "Episode Name": ep_name,
                "Prompt": info.prompt,
                "Expected LLM Response": info.expected_llm_response,
                "Disease": info.disease_name,
            })
        except Exception as e:
            logger.error("API error for %s: %s", ep_name, e)
            continue

    df = pd.DataFrame(records)
    output_path = GENERATED_DATA_DIR / "House_Diagnosis_structured_openai.xlsx"
    df.to_excel(output_path, index=False, engine="openpyxl")
    logger.info("Saved structured prompts to %s", output_path)
    return df

def generate_llm_responses(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each prompt in the DataFrame, call the LLM to get the actual response.

    Args:
        df: DataFrame with a 'Prompt' column.

    Returns:
        Updated DataFrame including 'actual_llm_response' and 'model_used'.
    """
    responses: List[str] = []
    models: List[str] = []

    for idx, row in df.iterrows():
        prompt = row["Prompt"]
        logger.info("Generating response for row %d", idx)
        try:
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "user", "content": prompt},
                ],
            )
            responses.append(completion.choices[0].message.content)
            models.append(getattr(completion, "model", "gpt-4o-mini"))
        except Exception as e:
            logger.error("Error at row %d: %s", idx, e)
            responses.append(f"Error: {e}")
            models.append(None)

    df["actual_llm_response"] = responses
    df["model_used"] = models

    output_path = GENERATED_DATA_DIR / "House_Diagnosis_with_actual_responses.xlsx"
    df.to_excel(output_path, index=False, engine="openpyxl")
    logger.info("Saved LLM responses to %s", output_path)
    return df

def normalize(text: str) -> str:
    """Lowercase and remove non-alphanumeric characters for cleaner comparison."""
    import re
    return re.sub(r'[^a-z0-9]', '', text.lower())

def is_close_match(expected: str, actual: str, threshold: float = 0.8) -> bool:
    """Check if expected disease closely matches any word/phrase in the actual response."""
    expected_norm = normalize(expected)
    actual_norm = normalize(actual)

    # Exact containment check
    if expected_norm in actual_norm:
        return True

    # Token-wise fuzzy comparison
    for word in actual.split():
        word_norm = normalize(word)
        if SequenceMatcher(None, expected_norm, word_norm).ratio() >= threshold:
            return True

    return False

def compute_accuracy(df: pd.DataFrame) -> float:
    correct_flags = []

    for idx, row in df.iterrows():
        disease = str(row["Disease"])
        response = str(row["actual_llm_response"])
        correct = is_close_match(disease, response)
        correct_flags.append(correct)

    df["is_correct"] = correct_flags
    accuracy = sum(correct_flags) / len(correct_flags) if correct_flags else 0
    logger.info("Accuracy (fuzzy matched): %.2f%% (%d/%d correct)", accuracy * 100, sum(correct_flags), len(correct_flags))

    # Save updated Excel with correctness column
    output_path = GENERATED_DATA_DIR / "House_Diagnosis_with_actual_responses_and_accuracy.xlsx"
    df.to_excel(output_path, index=False, engine="openpyxl")
    logger.info("Saved responses with accuracy flags to %s", output_path)

    return accuracy


def main():
    """Main entry point for script execution."""
    df = collect_episode_prompts()
    generate_llm_responses(df)
    compute_accuracy(df)

if __name__ == "__main__":
    main()
