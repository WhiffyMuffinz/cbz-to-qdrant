# CBZ Processor

CBZ file processor with Qdrant vector storage.

## Installation

```bash
pip install -r requirements.txt
cp .env.example .env
```

## Usage

Make sure that vllm and qdrant are both up:
e.g.
```bash
vllm serve Qwen/Qwen3-VL-Embedding-2B --runner pooling --max-model-len 32768 --gpu-memory-utilization 0.7 --port 7997
```
and
```bash
podman run --port 6333:6333 --name qdrant -v ./qdrant:/qdrant/storage -e QDRANT__TELEMETRY_DISABLED=true qdrant/qdrant:latest
```

```bash
# Process CBZ files from directory
PYTHONPATH=src python -m cbz_processor /path/to/your/cbz/files -c data/checkpoint.json

# Resume interrupted processing
PYTHONPATH=src python -m cbz_processor /path/to/your/cbz/files -c data/checkpoint.json

# Reset checkpoint and start fresh
PYTHONPATH=src python -m cbz_processor /path/to/your/cbz/files -c data/checkpoint.json --reset

# Dry run to see what would be processed
PYTHONPATH=src python -m cbz_processor /path/to/your/cbz/files --dry-run
```


## Environment Variables

```env
VLLM_ENDPOINT=http://localhost:7997
VLLM_MODEL=Qwen/Qwen3-VL-Embedding-2B
CBZ_CHUNK_SIZE=10
EMBEDDING_BATCH_SIZE=16
QDRANT_BATCH_SIZE=100
WORKER_PARALLELISM=4
QDRANT_HOST=localhost
QDRANT_PORT=6333
EMBEDDING_DIMENSION=2048
MAX_RETRIES=3
```


## Features

- **Resume**: Checkpoint-based recovery from interruptions
- **Parallel**: Multi-worker processing for CBZ files
- **Logging**: Structured JSON logs


## Monitoring

Checkpoints saved to `data/checkpoint.json` with:
- Total processed files
- Images extracted
- Embeddings generated
- Points inserted
- Current status

Logs saved to `logs/processor_YYYYMMDD_HHMMSS.log`
