# CPU image with the model baked in: cold starts never download 2 GB.
FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HOME=/models
WORKDIR /srv

# CPU-only torch is ~200 MB instead of ~2 GB with CUDA libraries
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install -r requirements.txt

ARG EMBEDDING_MODEL=BAAI/bge-m3
ENV EMBEDDING_MODEL=${EMBEDDING_MODEL}
# Download only what we use: the repos also hold ONNX / image files (several GB) that would fill the build disk.
RUN python -c "import os; from huggingface_hub import snapshot_download as d; d(os.environ['EMBEDDING_MODEL'], ignore_patterns=['onnx/*', '*.onnx', '*.onnx_data', 'imgs/*', '*.jpg', '*.png', 'openvino/*', '*.msgpack', '*.h5', '*.ot'])"
# Prove the baked copy loads with no network (also what happens at runtime: nothing is downloaded on start).
ENV HF_HUB_OFFLINE=1
RUN python -c "import os; from sentence_transformers import SentenceTransformer as S; S(os.environ['EMBEDDING_MODEL'])"

COPY app ./app
RUN useradd -m appuser && chown -R appuser /models
USER appuser
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
