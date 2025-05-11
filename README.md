# Board Game Manual LangGraph Agent (Dockerized)

This Streamlit application is an advanced Q&A agent for board game manuals. It uses LangGraph for orchestration, OpenAI for LLM and function calling, Chroma Cloud for vector storage, and can query BoardGameGeek. Deployment to Hugging Face Spaces is via a custom `Dockerfile`.

## Project Structure

-   `src/`: Contains all Python source code for the agent and Streamlit app.
    -   `streamlit_app.py`: The main Streamlit UI.
    -   `agent.py`: LangGraph agent definition and logic.
    -   `tools.py`: Definitions for callable tools (Chroma search, BGG search, Clarification).
-   `data/`: For PDF manuals and the `supported_games.txt` list.
-   `Dockerfile`: Defines the container environment for Hugging Face Spaces.
-   `ingest.py`: Script to process PDFs and populate Chroma Cloud (run locally).
-   `requirements.txt`: Python dependencies.

## Prerequisites

1.  **Accounts:** OpenAI API access, Chroma Cloud account, GitHub account, Hugging Face account.
2.  **Software:** Git, Conda (for local environment management).
3.  **Python:** Python 3.11 recommended for local development (matches Dockerfile).

## I. Chroma Cloud Setup

1.  **Sign Up/In:** Go to [cloud.trychroma.com](https://cloud.trychroma.com/).
2.  **Create Instance:** Create a new Chroma Cloud instance (cluster).
3.  **Note Connection Details:** From your instance dashboard, get:
    *   `CHROMA_SERVER_HOST`
    *   `CHROMA_SERVER_HTTP_PORT` (usually `8000`)
    *   `CHROMA_API_KEY` (client API key for your instance, if applicable).
    *   Note the `CHROMA_COLLECTION_NAME` you intend to use (default is `board_game_manuals`).

## II. Local Setup

1.  **Clone the Repository:**
    ```bash
    git clone <your-github-repo-url>
    cd board-game-manual-qa-langgraph
    ```

2.  **Create `data/` Directory & Add PDFs:**
    If it doesn't exist, create it.
    ```bash
    mkdir -p data
    ```
    **Place your board game manual PDF files (e.g., `Catan_Rules.pdf`) into this `data/` directory.**

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
    *   Copy `.env.example` to `.env` in the project root:
        ```bash
        cp .env.example .env
        ```
    *   Edit the `.env` file and fill in your credentials:
        *   `OPENAI_API_KEY`
        *   `CHROMA_API_KEY`
        *   `CHROMA_SERVER_HOST`
        *   `CHROMA_SERVER_HTTP_PORT`
        *   `CHROMA_COLLECTION_NAME` (e.g., `board_game_manuals`)
        *   `LANGGRAPH_AGENT_MODEL` (e.g., `gpt-4-turbo-preview`)
        *   `HF_TOKEN` (Hugging Face access token with write permissions for Spaces)
        *   `HF_SPACE_ID` (Your Hugging Face Space ID, e.g., `YourUsername/BoardGameAgent`)

## III. Data Ingestion (into Chroma Cloud)

This step processes your PDF manuals from the `data/` directory and stores their embeddings and text in Chroma Cloud. **Run this locally before the first deployment or whenever you add/change PDF manuals.**

1.  **Ensure Conda Environment is Active:** `conda activate ./venv`
2.  **Run the Ingestion Script:**
    *   The script uses `CHROMA_COLLECTION_NAME` from your `.env` file (or defaults to `board_game_manuals`).
    *   Ensure your PDF files are in the `data/` directory.
    ```bash
    python ingest.py --pdf_dir data
    ```
    *   To use a specific collection name overriding `.env` or default for this run:
    ```bash
    python ingest.py --pdf_dir data --collection_name your_custom_collection
    ```
    *   To clear an existing collection before ingesting (use with caution!):
    ```bash
    python ingest.py --pdf_dir data --collection_name board_game_manuals --clear
    ```
    *   This script will also create/update `data/supported_games.txt`.
3.  **Commit `data/supported_games.txt`:** This file is essential for the deployed app to list supported games.
    ```bash
    git add data/supported_games.txt
    git commit -m "Update list of supported game manuals"
    # You'll push this later with other code changes.
    ```

## IV. Local Development (Running the Agent UI)

1.  **Ensure Conda Environment is Active.**
2.  **Run the Streamlit App:** From the project root directory:
    ```bash
    streamlit run src/streamlit_app.py
    ```
3.  Open your browser to the URL provided by Streamlit (usually `http://localhost:8501`). Test thoroughly.

## V. Deployment to Hugging Face Spaces

1.  **GitHub Repository:**
    *   Ensure your project is a GitHub repository.
    *   Commit and push all your local changes: `Dockerfile`, `src/` directory contents, `requirements.txt`, `ingest.py`, `.github/workflows/deploy.yml`, and `data/supported_games.txt`.

2.  **Hugging Face Space Setup:**
    *   **Create a new Space on Hugging Face manually (Recommended for first time with Docker):**
        *   Go to [huggingface.co/new-space](https://huggingface.co/new-space).
        *   Owner: You or your organization.
        *   Space name: (This, with owner, forms your `HF_SPACE_ID`).
        *   **Space SDK: Select "Docker"**. Choose "No template" or a generic Docker template if offered.
        *   Visibility: Public or Private.
        *   Click `Create Space`.
    *   The GitHub Action (`huggingface-cli repo create ... --exist-ok`) will attempt to create it if it doesn't exist, but it's best if it's already a "Docker" type Space.

3.  **GitHub Secrets:**
    *   In your GitHub repository, go to `Settings` > `Secrets and variables` > `Actions`.
    *   Click `New repository secret` and add:
        *   `HF_TOKEN`: Your Hugging Face access token (must have `write` permissions).
        *   `HF_SPACE_ID`: The ID for your Hugging Face Space (e.g., `YourHFUsername/BoardGameAgent`).

4.  **Hugging Face Space Secrets:**
    *   Go to your Hugging Face Space `Settings` tab.
    *   Scroll to `Repository secrets` and click `New secret`. Add the following (these are read by your application code inside the Docker container):
        *   `OPENAI_API_KEY`
        *   `CHROMA_API_KEY`
        *   `CHROMA_SERVER_HOST`
        *   `CHROMA_SERVER_HTTP_PORT`
        *   `CHROMA_COLLECTION_NAME` (e.g., `board_game_manuals`)
        *   `LANGGRAPH_AGENT_MODEL` (e.g., `gpt-4-turbo-preview`)

5.  **Trigger Deployment:**
    *   Commit and push your changes to the `main` branch (or the branch specified in `deploy.yml`) of your GitHub repository.
    ```bash
    git add .
    git commit -m "Prepare for Dockerized HF Spaces deployment"
    git push origin main
    ```
    *   This will trigger the GitHub Actions workflow in `.github/workflows/deploy.yml`.
