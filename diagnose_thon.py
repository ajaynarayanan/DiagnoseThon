import os
import re
import logging
import pandas as pd
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv  
from typing import List, Tuple
from utils import openAIPayLoadHelper, parseOpenAIRespone
from constants import SYSTEM_PROMPT, HOUSE_EPISODE_TITLES, BASE_URL

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

if not OPENAI_API_KEY:
    logger.error("OpenAI API key is missing. Please add it to the .env file.")
    exit(1)

class ParsedObject:
    """Parses website content and extracts medical information."""
    
    def __init__(self, url: str, episode_name: str) -> None:
        self.url = url
        self.episode_name = episode_name
        self.parsed_contents = self.parse_url(self.url)
    
    def parse_url(self, url: str) -> str:
        """Fetches and parses the webpage content."""
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        return soup.get_text()

    def get_prompt(self) -> str:
        """Constructs the prompt for the LLM based on parsed contents."""
        prompt = (
            "Use details in INFORMATION for creating your LLM prompt, required medical answer, and the disease name.\n"
            "### INFORMATION ###\n"
            f"{self.parsed_contents}\n"
        )
        return prompt

def extract_disease_name(response: str) -> str:
    """Extracts the disease name from the LLM response."""
    match = re.search(r"Disease\s*:\s*(.+)", response, re.IGNORECASE)
    return match.group(1).strip() if match else "Unknown"

def main() -> None:
    logger.info("=== Collecting Data ===")
    parsed_objs: List[ParsedObject] = []
    data = []

    # Collect all URLs
    for episode_name in HOUSE_EPISODE_TITLES:
        url = f"{BASE_URL}{episode_name.replace(' ', '_')}"
        logger.info(f"Processing: {episode_name} ::: {url}")
        try:
            parsed_obj = ParsedObject(url, episode_name)
            parsed_objs.append(parsed_obj)
        except Exception as error:
            logger.error(f"Failed to process {url}: {error}")

    logger.info("=== Generating Medical Questions and Answers ===")

    # Call the API and collect results
    for parsed_obj in parsed_objs:
        prompt = parsed_obj.get_prompt()

        # Format API payload
        headers, payload = openAIPayLoadHelper(prompt, OPENAI_API_KEY, SYSTEM_PROMPT)

        # Make the POST API call
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        response.raise_for_status()

        parsed_output = parseOpenAIRespone(response)
        disease = extract_disease_name(parsed_output)
        
        # Append results to the data list
        data.append([parsed_obj.episode_name, prompt, parsed_output, disease])

    # Create a Pandas DataFrame
    df = pd.DataFrame(data, columns=["Episode Name", "Prompt", "Expected LLM Response", "Disease"])

    # Save to an Excel file
    excel_filename = "House_Diagnosis.xlsx"
    df.to_excel(excel_filename, index=False, engine="openpyxl")

    logger.info(f"Excel file '{excel_filename}' has been generated successfully!")

if __name__ == "__main__":
    main()
