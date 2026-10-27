# GenAI Deployment Lab — SmolLM2

## Project Overview

This project deploys the open-weight `HuggingFaceTB/SmolLM2-360M-Instruct` language model as a REST API using FastAPI and Google Cloud Run.

The application provides text-generation and health-check endpoints. The experiment evaluates response time, generated output length, output quality, and estimated serving cost.

## Model Details

- **Model:** SmolLM2-360M-Instruct
- **Model ID:** `HuggingFaceTB/SmolLM2-360M-Instruct`
- **Parameter count:** Approximately 360 million
- **License:** Apache 2.0
- **Inference:** CPU-only

## API Endpoints

- `GET /health` — checks application and model status.
- `POST /generate` — generates a response from a text prompt.
- `/docs` — interactive API documentation provided by FastAPI.

## Generation Settings

- Maximum new tokens: 128
- Temperature: 0.7

## Project Structure

- `app/main.py` — FastAPI application and model inference logic.
- `Dockerfile` — container build configuration.
- `pyproject.toml` and `uv.lock` — project dependencies and locked versions.
- `requirements.txt` — alternative dependency list.
- `screenshots/` — experiment and deployment evidence.
- `results/` — recorded experiment results, if applicable.

## Local Execution

Install `uv` and run the following commands from the project directory:

```powershell
uv sync --frozen
uv run uvicorn app.main:app --host 127.0.0.1 --port 8080
```

Then open `http://127.0.0.1:8080/docs` to inspect and test the API.

The first run may download model and tokenizer files from Hugging Face if they are not already cached.

## Deployment and Cleanup

The application was deployed to Google Cloud Run in the `asia-south1` region for the assignment experiment. The Cloud Run service, its container image, and the associated Artifact Registry repository were deleted after testing.

The deployment URL is no longer expected to be available.

## Experiment and Report

The accompanying Lab Assignment 7 report documents the model configuration, ten-prompt evaluation, response times, output lengths, observed output-quality issues, cost assumptions, deployment evidence, and cleanup verification.