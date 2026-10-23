# VERDANT ML API — portable image (Hugging Face Spaces, Render, Railway, Fly all honour $PORT)
FROM python:3.12-slim

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    PYTHONUNBUFFERED=1
WORKDIR /home/user/app

COPY --chown=user requirements-deploy.txt .
RUN pip install --no-cache-dir --user -r requirements-deploy.txt

# Bake the pretrained weights into the image so cold starts skip the download
COPY --chown=user download_models.py .
RUN python download_models.py

COPY --chown=user app ./app

EXPOSE 7860
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
