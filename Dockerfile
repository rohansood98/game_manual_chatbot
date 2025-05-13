FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    libxml2-dev libxslt1-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt ./
RUN pip3 install -r requirements.txt

# Copy application code and data
COPY src/ ./src/
COPY .streamlit/config.toml ./.streamlit/config.toml
COPY data/supported_games.txt ./data/supported_games.txt

EXPOSE 8501
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ENABLECORS=false
ENV PYTHONPATH=/app

CMD ["streamlit", "run", "src/streamlit_app.py", "--server.address=0.0.0.0"]