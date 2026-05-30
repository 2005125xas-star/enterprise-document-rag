# Public Real Document Demo Transcript

This file is a placeholder for a real-document demo transcript.

No public university policy PDFs or TXT files are committed by default. To generate a real transcript:

1. Verify the official source URLs in `data/public_eval/public_sources.yaml`.
2. Download only publicly available policy documents.
3. Place them in `data/public_docs/` using the listed `local_filename` values.
4. Run:

```bash
python -m src.evaluation.run_public_eval
```

Do not place private coursework, VLE/Moodle/Learning Mall/Canvas files, internal university documents, or copyrighted materials without permission in `data/public_docs/`.

MockProvider may be used for retrieval, citation, and refusal-path validation, but it does not represent real LLM answer quality. A real OpenAI-compatible provider should be used for qualitative answer review.
