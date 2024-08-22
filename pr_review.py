"""
This script performs an AI-powered review of GitHub pull requests.
It fetches PR details, generates a review, and outputs the result.
"""

import os
import json
import logging
from typing import Optional, Dict, Any
import requests
from dotenv import load_dotenv
from openai import AzureOpenAI
import instructor
from pydantic import BaseModel, Field, ValidationError, field_validator
import pydantic  # Add this line
import jsonschema
import tiktoken
import time
from halo import Halo
import backoff

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

# Load environment variables
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')
AZURE_OPENAI_API_KEY = os.getenv('AZURE_OPENAI_API_KEY')
AZURE_OPENAI_ENDPOINT = os.getenv('AZURE_OPENAI_ENDPOINT')
AZURE_OPENAI_API_VERSION = os.getenv('AZURE_OPENAI_API_VERSION')
PR_NUMBER = int(os.getenv('PR_NUMBER', 0))
REPO_FULL_NAME = os.getenv('REPO_FULL_NAME')
SCORE_CARD_PATH = os.getenv('SCORE_CARD_PATH', 'score-card.json')

if not all([GITHUB_TOKEN, AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_VERSION, PR_NUMBER, REPO_FULL_NAME]):
    logging.error("One or more required environment variables are missing. Exiting.")
    exit(1)

# Set up Azure OpenAI client
azure_openai_client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_OPENAI_API_VERSION
)

# Set up Instructor client
instructor_client = instructor.patch(azure_openai_client)

# Define Pydantic models
class ScoreValue(BaseModel):
    value: Optional[int] = None
    description: Optional[str] = None

class Score(BaseModel):
    Excellent: Optional[ScoreValue] = None
    Some_issues: Optional[ScoreValue] = None
    Unacceptable: Optional[ScoreValue] = None
    n_a: Optional[dict] = Field(None, alias='n/a')  # Change this line

    @field_validator('n_a', mode='before')
    @classmethod
    def set_n_a_default(cls, v):
        return v or {"description": "Not applicable"}

class Parameter(BaseModel):
    name: str
    explanation: str
    category: str
    score: Optional[Score] = None

class PRReview(BaseModel):
    parameters: list[Parameter] = Field(description="List of parameters to evaluate the PR")
    summary: str = Field(description="A brief summary of the changes in the PR")
    total_score: int = Field(description="Total score of the PR review")
    recommendations: list[str] = Field(description="List of recommendations for improving the PR")

def count_tokens(text: str) -> int:
    """Count the number of tokens in a given text."""
    encoding = tiktoken.encoding_for_model("gpt-4o")
    return len(encoding.encode(text))

@backoff.on_exception(backoff.expo, requests.exceptions.RequestException, max_tries=3)
def get_pr_files(repo_full_name: str, pr_number: int) -> Dict[str, str]:
    """Fetch the files changed in a pull request."""
    url = f"https://api.github.com/repos/{repo_full_name}/pulls/{pr_number}/files"
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json"
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    files = response.json()

    pr_files = {}
    for file in files:
        filename = file['filename']
        if 'patch' in file:
            pr_files[filename] = file['patch']
        else:
            # If 'patch' is not available, we'll use a placeholder
            pr_files[filename] = f"[File content not available for {filename}]"

        # Optionally, you can add more file information
        # pr_files[filename] = {
        #     'patch': file.get('patch', '[Content not available]'),
        #     'status': file['status'],
        #     'additions': file['additions'],
        #     'deletions': file['deletions'],
        #     'changes': file['changes'],
        # }

    return pr_files

def construct_review_prompt(pr_files: Dict[str, str], score_card: dict) -> str:
    """Construct the prompt for the AI to review the pull request."""
    context = "\n".join([f"File: {filename}\nChanges:\n{content}" for filename, content in pr_files.items()])
    prompt = f"""
    Review the following pull request changes and provide a structured review based on the given score card:

    Score Card:
    {json.dumps(score_card, indent=2)}

    PR Content:
    {context}

    Provide a review that includes:
    1. An evaluation for each parameter in the score card. Never skip any parameter.
    2. A brief summary of the changes.
    3. A list of recommendations for improving the PR.
    """
    return prompt

def generate_pr_review(prompt: str) -> PRReview:
    """Generate a pull request review using the AI model."""
    start_time = time.time()
    spinner = Halo(text='Generating PR review', spinner='dots')
    spinner.start()

    try:
        response = instructor_client.chat.completions.create(
            model="gpt-4o-mini",
            response_model=PRReview,
            messages=[
                {"role": "system", "content": """
                You are an AI assistant that reviews pull requests for the Daytona Content Programme.
                Provide a review with parameters, summary, total_score, and recommendations.
                For each parameter, provide a score with the following structure:
                {
                    "Excellent": {"value": 5, "description": "..."},
                    "Some_issues": {"value": 3, "description": "..."},
                    "Unacceptable": {"value": 1, "description": "..."},
                    "n/a": {"description": "Not applicable"}
                }
                Choose the most appropriate score for each parameter and set the others to null.
                If a parameter is not applicable, set "n/a" to {"description": "Not applicable"} and the others to null.
                """},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        elapsed_time = time.time() - start_time
        spinner.succeed(f"Successfully generated PR review in {elapsed_time:.2f} seconds")
        print(f"Review: {response}")
        return response
    except instructor.exceptions.InstructorRetryException as e:
        spinner.fail(f"Error generating PR review after retries: {str(e)}")
        if hasattr(e, '__cause__') and isinstance(e.__cause__, pydantic.ValidationError):
            print(f"Validation errors: {e.__cause__.errors()}")
            print(f"Full error: {e.__cause__}")
            # Try to access the raw response
            if hasattr(e, 'response'):
                print(f"Raw response: {e.response}")
        raise
    except Exception as e:
        spinner.fail(f"Unexpected error in generate_pr_review: {str(e)}")
        print(f"Full error: {e}")
        raise
    finally:
        spinner.stop()

def validate_review(review: PRReview, schema_path: str) -> bool:
    """Validate the generated review against the schema."""
    try:
        with open(schema_path, 'r') as schema_file:
            schema = json.load(schema_file)
        jsonschema.validate(instance=review.dict(), schema=schema)
        return True
    except jsonschema.exceptions.ValidationError as e:
        logging.error(f"Review validation failed: {e}")
        logging.error(f"Failed validating {e.validator} in schema: {json.dumps(e.schema, indent=2)}")
        logging.error(f"On instance: {json.dumps(e.instance, indent=2)}")
        logging.error(f"Path of error: {' -> '.join(str(path) for path in e.path)}")
        return False

def format_review_comment(review: PRReview) -> str:
    """Format the review as a comment to be posted on GitHub."""
    comment = "# AI PR Review\n\n"
    comment += f"## Summary\n{review.summary}\n\n"
    comment += f"## Total Score: {review.total_score}\n\n"
    comment += "## Detailed Evaluation\n"
    for param in review.parameters:
        comment += f"### {param.name}\n"
        comment += f"Category: {param.category}\n"
        comment += f"Explanation: {param.explanation}\n"
        comment += f"Score: {param.score.dict()}\n\n"
    comment += "## Recommendations\n"
    for recommendation in review.recommendations:
        comment += f"- {recommendation}\n"
    return comment

def main():
    """Main function to orchestrate the PR review process."""
    try:
        print(f"Starting PR review for PR #{PR_NUMBER} in repository {REPO_FULL_NAME}")

        with Halo(text='Fetching PR files', spinner='dots') as spinner:
            pr_files = get_pr_files(REPO_FULL_NAME, PR_NUMBER)
            spinner.succeed(f"Retrieved {len(pr_files)} files from the PR")

        with Halo(text='Loading score card', spinner='dots') as spinner:
            with open(SCORE_CARD_PATH, 'r') as score_card_file:
                score_card = json.load(score_card_file)
            spinner.succeed(f"Loaded score card from {SCORE_CARD_PATH}")

        with Halo(text='Constructing review prompt', spinner='dots') as spinner:
            prompt = construct_review_prompt(pr_files, score_card)
            spinner.succeed("Constructed review prompt")

        try:
            review = generate_pr_review(prompt)
        except instructor.exceptions.InstructorRetryException as e:
            print("Failed to generate PR review due to validation errors")
            print(f"Error details: {e}")
            return {"status": "error", "pr_number": PR_NUMBER, "message": "Failed to generate valid PR review", "details": str(e)}

        with Halo(text='Validating review', spinner='dots') as spinner:
            if validate_review(review, 'schema/score-card.schema.json'):
                spinner.succeed("Review validation successful")
                comment = format_review_comment(review)
                with open('ai_review_result.md', 'w') as f:
                    f.write(comment)
                print(f"AI PR review completed successfully for PR #{PR_NUMBER}.")
                return {"status": "success", "pr_number": PR_NUMBER, "message": "Review generated successfully"}
            else:
                spinner.fail("Review validation failed")
                error_message = "Failed to generate a valid PR review."
                with open('ai_review_result.md', 'w') as f:
                    f.write(error_message)
                return {"status": "error", "pr_number": PR_NUMBER, "message": error_message}
    except Exception as e:
        error_message = f"An unexpected error occurred: {str(e)}"
        print(error_message)
        print(f"Full error: {e}")
        logging.exception("Detailed traceback:")
        return {"status": "error", "pr_number": PR_NUMBER, "message": error_message}

if __name__ == "__main__":
    main()