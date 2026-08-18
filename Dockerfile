FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the AI model into the image at build time so the container works
# offline at runtime and the demo doesn't eat a download on first request.
RUN python -c "from transformers import pipeline; pipeline('image-classification', model='prithivMLmods/deepfake-detector-model-v1')"

COPY app.py db.py forensics.py ./
COPY templates/ templates/
COPY static/ static/

RUN mkdir -p instance/media

EXPOSE 8000

CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "1", "--threads", "4", "--timeout", "300", "app:app"]
