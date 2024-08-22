# AI PR Review

AI PR Review is an automated system that uses Azure OpenAI to review pull requests for the Daytona Content Programme. It provides structured feedback based on a predefined score card, helping to maintain high-quality content contributions.

## Table of Contents

- [AI PR Review](#ai-pr-review)
  - [Table of Contents](#table-of-contents)
  - [Features](#features)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Configuration](#configuration)
  - [Usage](#usage)
  - [How It Works](#how-it-works)
  - [Contributing](#contributing)
  - [License](#license)

## Features

- Automated PR reviews triggered by GitHub Actions
- Customizable score card for content evaluation
- Integration with Azure OpenAI for intelligent content analysis
- Structured feedback posted as PR comments
- JSON schema validation for review consistency

## Prerequisites

- Python 3.9+
- GitHub repository with Actions enabled
- Azure OpenAI API access
- GitHub Personal Access Token with repo scope

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/your-username/ai-pr-review.git
   cd ai-pr-review
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up your development environment using the provided `devcontainer.json` file (optional, requires VS Code with Remote - Containers extension).

## Configuration

1. Create a `.env` file in the project root and add the following environment variables:
   ```
   AZURE_OPENAI_API_KEY=your_azure_openai_api_key
   AZURE_OPENAI_ENDPOINT=your_azure_openai_endpoint
   AZURE_OPENAI_API_VERSION=your_azure_openai_api_version
   GITHUB_TOKEN=your_github_personal_access_token
   ```

2. Customize the `score-card.json` file to define your content evaluation criteria.

3. Update the GitHub Actions workflow file (`.github/workflows/pr-review.yml`) if necessary.

## Usage

The AI PR Review system is automatically triggered when a new pull request is opened in your GitHub repository. It will:

1. Analyze the content of the PR
2. Generate a review based on the defined score card
3. Post the review as a comment on the PR

To manually run the review process locally:

```
python scripts/ai_pr_review.py
```

## How It Works

1. When a new PR is opened, the GitHub Action is triggered.
2. The action runs the `ai_pr_review.py` script.
3. The script fetches the PR content using the GitHub API.
4. It then sends the content to Azure OpenAI for analysis.
5. The AI generates a structured review based on the score card.
6. The review is validated against the JSON schema.
7. If valid, the review is posted as a comment on the PR.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.