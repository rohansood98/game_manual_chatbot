FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    libxml2-dev libxslt1-dev \
    # Add any other system-level dependencies your tools might need
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker layer caching
COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the rest of the application code and data needed at runtime
COPY src/ ./src/
COPY data/ ./data/ # Copies the data directory (containing supported_games.txt) into /app/data/

# Environment variables for the app (can be overridden by Hugging Face secrets)
# ENV OPENAI_API_KEY="" # Set in HF secrets
# ENV CHROMA_SERVER_HOST="" # Set in HF secrets
# ... etc.

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "src/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]