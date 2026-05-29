from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from statistics import mean

from src.evaluation.metrics import hit_at_k, reciprocal_rank
from src.models import EvaluationExample, EvaluationResult
from src.qa.pipeline import QAPipeline


def evaluate(pipeline: QAPipeline, examples: list[EvaluationExample], top_k: int = 5) -> EvaluationResult:
    details: list[dict] = []
    answerable_details: list[dict] = []
    unanswerable_details: list[dict] = []

    for example in examples:
        answer = pipeline.answer_question(example.question, top_k=top_k)
        retrieved_ids = [result.chunk.chunk_id for result in answer.retrieval_results]
        retrieved_documents = [result.chunk.document_id for result in answer.retrieval_results]
        top_score = answer.retrieval_results[0].score if answer.retrieval_results else None
        item = {
            "question": example.question,
            "answerable": example.answerable,
            "is_answered": answer.is_answered,
            "returned_no_answer": not answer.is_answered,
            "latency_ms": answer.latency_ms,
            "latency": answer.latency_ms,
            "answer": answer.answer,
            "retrieved_chunk_ids": retrieved_ids,
            "retrieved_document_ids": retrieved_documents,
            "expected_chunk_ids": example.expected_chunk_ids,
            "expected_document_ids": example.expected_document_ids,
            "expected_doc": "|".join(example.expected_document_ids),
            "expected_page": example.expected_page,
            "top_score": top_score,
            "hit_at_3": hit_at_k(answer.retrieval_results, example.expected_chunk_ids, example.expected_document_ids, 3),
            "hit_at_5": hit_at_k(answer.retrieval_results, example.expected_chunk_ids, example.expected_document_ids, 5),
            "reciprocal_rank": reciprocal_rank(
                answer.retrieval_results,
                example.expected_chunk_ids,
                example.expected_document_ids,
            ),
            "no_answer_correct": (not answer.is_answered) if not example.answerable else None,
        }
        details.append(item)
        if example.answerable:
            answerable_details.append(item)
        else:
            unanswerable_details.append(item)

    return EvaluationResult(
        hit_at_3=_mean_metric(answerable_details, "hit_at_3"),
        hit_at_5=_mean_metric(answerable_details, "hit_at_5"),
        mrr=_mean_metric(answerable_details, "reciprocal_rank"),
        no_answer_accuracy=_mean_metric(unanswerable_details, "no_answer_correct"),
        average_latency_ms=round(mean([item["latency_ms"] for item in details]), 2) if details else 0.0,
        total_examples=len(examples),
        details=details,
    )


def load_examples(path: str | Path) -> list[EvaluationExample]:
    source = Path(path)
    if source.suffix.lower() == ".json":
        data = json.loads(source.read_text(encoding="utf-8"))
        return [_example_from_mapping(item) for item in data]
    if source.suffix.lower() == ".csv":
        with source.open("r", encoding="utf-8", newline="") as handle:
            return [_example_from_mapping(row) for row in csv.DictReader(handle)]
    raise ValueError("Evaluation examples must be CSV or JSON.")


def save_evaluation_outputs(
    result: EvaluationResult,
    csv_path: str | Path = "outputs/eval_results.csv",
    report_path: str | Path = "outputs/evaluation_report.md",
) -> None:
    csv_target = Path(csv_path)
    report_target = Path(report_path)
    csv_target.parent.mkdir(parents=True, exist_ok=True)
    report_target.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "question",
        "answerable",
        "expected_doc",
        "expected_page",
        "retrieved_chunk_ids",
        "top_score",
        "hit_at_3",
        "hit_at_5",
        "reciprocal_rank",
        "returned_no_answer",
        "latency",
        "answer",
    ]
    with csv_target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in result.details:
            writer.writerow({key: _format_csv_value(item.get(key)) for key in fieldnames})

    report_target.write_text(_render_report(result), encoding="utf-8")


def _mean_metric(items: list[dict], key: str) -> float:
    values = [float(item[key]) for item in items if item.get(key) is not None]
    return round(mean(values), 4) if values else 0.0


def _example_from_mapping(item: dict) -> EvaluationExample:
    return EvaluationExample(
        question=str(item["question"]),
        answerable=_to_bool(item.get("answerable", True)),
        expected_chunk_ids=_split_ids(item.get("expected_chunk_ids", "")),
        expected_document_ids=_split_ids(item.get("expected_document_ids") or item.get("expected_doc", "")),
        expected_page=_to_optional_int(item.get("expected_page")),
        expected_answer=item.get("expected_answer") or None,
    )


def _split_ids(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in str(value).split("|") if item.strip()]


def _to_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _format_csv_value(value: object) -> object:
    if isinstance(value, list):
        return "|".join(str(item) for item in value)
    return value


def _to_optional_int(value: object) -> int | None:
    if value is None or str(value).strip() == "":
        return None
    return int(str(value).strip())


def _render_report(result: EvaluationResult) -> str:
    metrics = asdict(result)
    details = metrics.pop("details")
    interpretation = _interpret_result(result)
    lines = [
        "# Enterprise Document RAG QA System - Evaluation Report",
        "",
        "## Summary",
        "",
        f"- Project name: Enterprise Document RAG QA System",
        f"- Number of evaluation questions: {result.total_examples}",
        f"- Hit@3: {result.hit_at_3:.4f}",
        f"- Hit@5: {result.hit_at_5:.4f}",
        f"- MRR: {result.mrr:.4f}",
        f"- No-answer accuracy: {result.no_answer_accuracy:.4f}",
        f"- Average latency (ms): {result.average_latency_ms:.2f}",
        "",
        "## Interpretation",
        "",
        interpretation,
        "",
        "## Known Limitations",
        "",
        "- The default evaluation fixture is intentionally small and is meant as a smoke test, not a benchmark.",
        "- DOCX page numbers are approximated as page 1 because DOCX files do not store stable rendered page breaks.",
        "- MockProvider validates the local pipeline deterministically but does not measure final LLM answer quality.",
        "- Hashing embeddings are used by the evaluation CLI to avoid external downloads; the app uses sentence-transformers when available.",
        "",
        "## Example Results",
        "",
    ]
    for item in details:
        lines.extend(
            [
                f"### {item['question']}",
                "",
                f"- Answerable: {item['answerable']}",
                f"- Answered: {item['is_answered']}",
                f"- Retrieved chunks: {', '.join(item['retrieved_chunk_ids']) or 'None'}",
                f"- Answer: {item['answer']}",
                "",
            ]
        )
    return "\n".join(lines)


def _interpret_result(result: EvaluationResult) -> str:
    if result.total_examples == 0:
        return "No evaluation examples were provided."
    if result.hit_at_3 >= 0.8 and result.no_answer_accuracy >= 0.8:
        return "The smoke evaluation indicates that retrieval and no-answer handling are working on the included fixture."
    return "The smoke evaluation surfaced quality gaps. Inspect per-question results before treating this configuration as portfolio-ready."
