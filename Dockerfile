# RedFlag Lens: API (port 8000) + dashboard (port 8501). Models are baked in; GEMINI_API_KEY is optional
# (without it the summary falls back to the template built from findings).
FROM python:3.13-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
COPY src ./src
COPY models ./models
COPY data/annotations/guide.md ./data/annotations/guide.md
ENV PYTHONPATH=/app REDFLAG_MODEL=finbert
EXPOSE 8000 8501
CMD ["sh", "-c", "uvicorn src.app.api:app --host 0.0.0.0 --port 8000 & streamlit run src/app/ui.py --server.port 8501 --server.address 0.0.0.0"]
