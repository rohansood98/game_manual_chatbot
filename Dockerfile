FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    libxml2-dev libxslt1-dev \
    # Add other system deps if needed by preprocessing libs
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt ./
RUN pip3 install -r requirements.txt
# Optional: Download NLTK data here if using nltk for preprocessing
# RUN python3 -m nltk.downloader punkt stopwords wordnet averaged_perceptron_tagger # Example

# Copy application code and data
COPY src/ ./src/
COPY .streamlit/config.toml ./.streamlit/config.toml
# Makes data/supported_games.txt available at /app/data/
COPY data/supported_games.txt ./data/supported_games.txt

EXPOSE 8501
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ENABLECORS=false
ENV PYTHONPATH=/app
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

CMD ["streamlit", "run", "src/streamlit_app.py", "--server.address=0.0.0.0"]