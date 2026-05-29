# Enterprise Document RAG QA System - Evaluation Report

## Summary

- Project name: Enterprise Document RAG QA System
- Number of evaluation questions: 3
- Hit@3: 1.0000
- Hit@5: 1.0000
- MRR: 1.0000
- No-answer accuracy: 1.0000
- Average latency (ms): 0.13

## Interpretation

The smoke evaluation indicates that retrieval and no-answer handling are working on the included fixture.

## Known Limitations

- The default evaluation fixture is intentionally small and is meant as a smoke test, not a benchmark.
- DOCX page numbers are approximated as page 1 because DOCX files do not store stable rendered page breaks.
- MockProvider validates the local pipeline deterministically but does not measure final LLM answer quality.
- Hashing embeddings are used by the evaluation CLI to avoid external downloads; the app uses sentence-transformers when available.

## Example Results

### How long are customer records retained?

- Answerable: True
- Answered: True
- Retrieved chunks: sample-policy:p1:c1
- Answer: Based on the uploaded documents, Enterprise Customer Records Policy

Customer records must be retained for seven years after account closure. [1]

### How are backups protected?

- Answerable: True
- Answered: True
- Retrieved chunks: sample-policy:p1:c1
- Answer: Based on the uploaded documents, Backup Security Standard

Operational backups are encrypted at rest and reviewed every quarter. [1]

### quantum asteroid orbit

- Answerable: False
- Answered: False
- Retrieved chunks: None
- Answer: I could not find enough evidence in the uploaded documents.
