# Ticket Sentiment Hybrid

Python-only ticket sentiment analysis for `negative`, `neutral`, and `positive` labels.

This project uses a hybrid pipeline instead of relying on one model:

- text normalization for casual and informal English
- a domain lexicon for support-ticket phrasing
- retrieval over labeled ticket examples for RAG-style context
- an optional transformer sentiment model
- an optional LLM arbiter for ambiguous cases

## Important note on accuracy

No sentiment system can honestly guarantee `100%` accuracy on real ticket traffic. The code here is built to be strong, extensible, and practical, but the final accuracy depends on:

- how representative your labeled ticket history is
- whether you install and enable the full ML stack
- whether you tune the retrieval examples and thresholds on your own data
- whether you evaluate on a clean held-out dataset

To get the best results, replace or expand `data/reference_tickets.jsonl` with your real labeled tickets and run the evaluation command on a validation set.

## Install

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

If you want to run with fewer dependencies first, the code still has pure-Python fallbacks for lexicon scoring and token-overlap retrieval, but accuracy will be lower than the full stack.

## Usage

Single prediction:

```bash
python -m ticket_sentiment predict --text "pls fix this, app is kinda broken rn" --json
```

Batch prediction from CSV or JSONL:

```bash
python -m ticket_sentiment batch --input sample_tickets.csv --output predictions.jsonl
```

Evaluate on labeled data:

```bash
python -m ticket_sentiment evaluate --input validation.csv
```

Add a new retrieval example:

```bash
python -m ticket_sentiment add-example --label negative --text "Still cannot access the dashboard and users are frustrated."
```

Run the API:

```bash
python -m ticket_sentiment serve --host 127.0.0.1 --port 8000
```

Then call:

```bash
curl -X POST http://127.0.0.1:8000/predict ^
  -H "Content-Type: application/json" ^
  -d "{\"text\":\"Thanks, the issue is fixed now\"}"
```

## LLM options

The LLM layer is optional and is only used when the ensemble is uncertain, unless you force it on.

### OpenAI

```bash
set OPENAI_API_KEY=your_key_here
set TICKET_SENTIMENT_OPENAI_MODEL=gpt-4.1-mini
set TICKET_SENTIMENT_LLM_PROVIDER=openai
```

### Ollama

```bash
set TICKET_SENTIMENT_LLM_PROVIDER=ollama
set TICKET_SENTIMENT_OLLAMA_MODEL=llama3.1:8b-instruct
set TICKET_SENTIMENT_OLLAMA_BASE_URL=http://localhost:11434
```

## Project structure

- `ticket_sentiment/preprocessing.py`: slang and informal-English normalization
- `ticket_sentiment/lexicon.py`: support-domain rules
- `ticket_sentiment/retrieval.py`: retrieval backend and RAG vote
- `ticket_sentiment/models.py`: transformer and VADER wrappers
- `ticket_sentiment/llm.py`: optional LLM arbitration
- `ticket_sentiment/service.py`: end-to-end orchestration
- `data/reference_tickets.jsonl`: starter labeled examples

## Recommended next step for higher accuracy

Use your own ticket history.

Even a few hundred carefully labeled historical tickets from your actual domain usually improve the RAG layer and final accuracy much more than swapping model names randomly.

