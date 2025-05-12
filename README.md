# Board Game Manual LangGraph Agent (Qdrant + Dockerized)

This Streamlit application is an advanced Q&A agent for board game manuals. It uses:
-   LangGraph for orchestration.
-   OpenAI for LLM reasoning and function calling.
-   Qdrant Cloud for vector storage and retrieval.
-   BoardGameGeek API for external lookups.
-   Text preprocessing for cleaner data ingestion.
-   Docker for deployment on Hugging Face Spaces.

## Project Structure

-   `src/`: Contains Python source code (Streamlit app, LangGraph agent, tools).
-   `data/`: Stores input PDF manuals and the generated `supported_games.txt`.
-   `Dockerfile`: Defines the container environment for deployment.
-   `ingest.py`: Script to process PDFs and populate Qdrant Cloud (run locally).
-   `requirements.txt`: Python dependencies.
-   `.env.example`: Template for environment variables.

## Prerequisites

1.  **Accounts:** OpenAI API access, Qdrant Cloud account, GitHub account, Hugging Face account.
2.  **Software:** Git, Conda.
3.  **Python:** Python 3.11 recommended locally (matches `Dockerfile`).

## I. Qdrant Cloud Setup

1.  **Sign Up/In:** Create an account and set up a cluster on [cloud.qdrant.io](https://cloud.qdrant.io/).
2.  **Get Credentials:** From your Qdrant Cloud dashboard, find:
    *   Your cluster **URL** (e.g., `https://<your-id>.<region>.cloud.qdrant.io:6333`).
    *   An **API Key** (create one under Access Management if needed).
3.  **Note Collection Name:** Decide on a collection name (default is `board_game_manuals`). The `ingest.py` script will create it if it doesn't exist.

## II. Local Setup

1.  **Clone Repository:**
    ```bash
    git clone <your-github-repo-url>
    cd board-game-manual-qa-langgraph # Navigate into the project directory
    ```

2.  **Create `data/` Directory & Add PDFs:**
    ```bash
    mkdir -p data
    ```
    Place your board game manual PDF files into the `data/` directory. Can use donwloan_manuals.py to get some manuals.

3.  **Create Conda Environment (Python 3.11):**
    ```bash
    conda create -p ./venv python=3.11 -y
    conda activate ./venv
    ```

4.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

5.  **Set Up Environment Variables (`.env` file):**
    *   Copy `.env.example` to `.env` in the project root: `cp .env.example .env`
    *   Edit `.env` and fill in **your actual credentials**:
        *   `OPENAI_API_KEY`
        *   `QDRANT_URL`
        *   `QDRANT_API_KEY` (leave blank if your cluster has no auth)
        *   `QDRANT_COLLECTION_NAME` (e.g., `board_game_manuals`)
        *   `LANGGRAPH_AGENT_MODEL` (e.g., `gpt-4-turbo-preview`)
        *   `HF_TOKEN` (Hugging Face token for deployment)
        *   `HF_SPACE_ID` (Your HF Space ID, e.g., `YourUsername/BoardGameAgent`)

## III. Data Ingestion (into Qdrant Cloud)

This step processes your local PDF manuals, preprocesses the text, generates embeddings, and stores them in your Qdrant Cloud collection. **Run this locally before the first deployment or whenever you add/change PDF manuals.**

1.  **Ensure Conda Environment is Active:** `conda activate ./venv`
2.  **Run the Ingestion Script:**
    *   Uses settings from your `.env` file.
    *   Ensure PDFs are in `data/`.
    ```bash
    python ingest.py --pdf_dir data
    ```
    *   To explicitly set collection name for this run:
    ```bash
    python ingest.py --pdf_dir data --collection_name your_custom_collection
    ```
    *   **To clear and recreate the collection** before ingesting:
    ```bash
    python ingest.py --pdf_dir data --collection_name board_game_manuals --clear
    ```
    *   This script updates `data/supported_games.txt`.
3.  **Commit `data/supported_games.txt`:** This file is copied into the Docker image and used by the app. **It's crucial to commit this file after running ingest.**
    ```bash
    git add data/supported_games.txt
    git commit -m "Update list of supported game manuals after ingestion"
    # Push later with other code changes
    ```

## IV. Local Development (Running the Agent UI)

1.  **Ensure Conda Environment is Active.**
2.  **Run the Streamlit App:** From the project root directory:
    ```bash
    streamlit run src/streamlit_app.py
    ```
3.  Open in browser (usually `http://localhost:8501`). Test the chat, tool usage, and clarification flow.

## V. Deployment to Hugging Face Spaces

1.  **GitHub Repository:**
    *   Ensure project is a GitHub repo.
    *   Commit and push all local changes: `Dockerfile`, `src/`, `requirements.txt`, `ingest.py`, `.github/workflows/deploy.yml`, `data/supported_games.txt`.

2.  **Hugging Face Space Setup:**
    *   **Create a new Space on Hugging Face (Recommended):**
        *   SDK: **Docker** (No template or basic).
        *   Name/Owner forms `HF_SPACE_ID`.
    *   The GitHub Action will try to create it, but ensuring it's "Docker" type beforehand is best.

3.  **GitHub Secrets:**
    *   In GitHub repo `Settings > Secrets and variables > Actions`:
        *   `HF_TOKEN`
        *   `HF_SPACE_ID`

4.  **Hugging Face Space Secrets:**
    *   In your HF Space `Settings > Repository secrets`:
        *   `OPENAI_API_KEY`
        *   `QDRANT_URL`
        *   `QDRANT_API_KEY` (Provide even if empty string, if your client expects it)
        *   `QDRANT_COLLECTION_NAME`
        *   `LANGGRAPH_AGENT_MODEL`

5.  **Trigger Deployment:** Push changes to the `main` branch on GitHub.
    ```bash
    git add .
    git commit -m "Ready for Qdrant Dockerized HF deployment"
    git push origin main
    ```

6.  **Monitor Deployment:** Check GitHub Actions. Then monitor the Docker build process in your HF Space logs. The first build can take several minutes.

Your deployed agent will now use Qdrant Cloud for its knowledge base retrieval.