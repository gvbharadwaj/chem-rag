# chem-rag

Multi-agent RAG demo over the Chemicals in Cosmetics dataset.
Returns an answer, evidence (CDPHId / CSFId / ChemicalId), query plan and warnings.

## Run

```
pip install -r requirements.txt
python main.py "Which products contain CAS 75-07-0?"
python -m pytest -q
```

Put the CSV at `data/for_assessment.csv`.

Optional LLM answers (Gemini, OpenAI-compatible): set `LLM_API_KEY`, `LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/`, `LLM_MODEL=gemini-flash-latest`. Without a key it uses a template.

## How it works

planner -> extraction -> semantic match -> SQL (DuckDB) -> synthesizer

- `chemrag/agents.py` - the 5 agents
- `chemrag/orchestrator.py` - routing + safety checks
- `chemrag/data.py` - load and clean (dedupe, fix CAS numbers)
