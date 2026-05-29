from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

from src.evaluation.metrics import hit_at_k, reciprocal_rank
from src.models import EvaluationExample, EvaluationResult
from src.qa.pipeline import QAPipeline, validate_citations


REQUIRED_EVAL_COLUMNS = {
    "question_id",
    "question",
    "question_type",
    "expected_doc",
    "expected_page",
    "expected_keywords",
    "answerable",
    "difficulty",
    "notes",
}


def evaluate(pipeline: QAPipeline, examples: list[EvaluationExample], top_k: int = 5) -> EvaluationResult:
    details: list[dict] = []
    answerable_details: list[dict] = []
    unanswerable_details: list[dict] = []

    for example in examples:
        answer = pipeline.answer_question(example.question, top_k=top_k)
        retrieved_ids = [result.chunk.chunk_id for result in answer.retrieval_results]
        retrieved_docs = [result.chunk.document_id for result in answer.retrieval_results]
        retrieved_pages = [result.chunk.page_number for result in answer.retrieval_results]
        top_score = answer.retrieval_results[0].score if answer.retrieval_results else None
        citation_present = answer.is_answered and validate_citations(answer.answer, answer.citations)
        item = {
            "question_id": example.question_id,
            "question": example.question,
            "question_type": example.question_type,
            "difficulty": example.difficulty,
            "answerable": example.answerable,
            "is_answered": answer.is_answered,
            "returned_no_answer": not answer.is_answered,
            "latency_ms": answer.latency_ms,
            "latency": answer.latency_ms,
            "answer": answer.answer,
            "answer_preview": _preview(answer.answer),
            "retrieved_chunk_ids": retrieved_ids,
            "retrieved_docs": retrieved_docs,
            "retrieved_document_ids": retrieved_docs,
            "retrieved_pages": retrieved_pages,
            "expected_chunk_ids": example.expected_chunk_ids,
            "expected_document_ids": example.expected_document_ids,
            "expected_doc": "|".join(example.expected_document_ids),
            "expected_page": example.expected_page,
            "expected_keywords": example.expected_keywords,
            "top_score": top_score,
            "hit_at_3": hit_at_k(answer.retrieval_results, example.expected_chunk_ids, example.expected_document_ids, 3),
            "hit_at_5": hit_at_k(answer.retrieval_results, example.expected_chunk_ids, example.expected_document_ids, 5),
            "reciprocal_rank": reciprocal_rank(
                answer.retrieval_results,
                example.expected_chunk_ids,
                example.expected_document_ids,
            ),
            "no_answer_correct": (not answer.is_answered) if not example.answerable else None,
            "citation_present": citation_present,
        }
        item["failure_reason"] = determine_failure_reason(example, item)
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
        answerable_count=len(answerable_details),
        unanswerable_count=len(unanswerable_details),
        citation_rate=_citation_rate(details),
        metrics_by_question_type=_group_metrics(details, "question_type"),
        metrics_by_difficulty=_group_metrics(details, "difficulty"),
        details=details,
    )


def load_examples(path: str | Path) -> list[EvaluationExample]:
    source = Path(path)
    if source.suffix.lower() == ".json":
        data = json.loads(source.read_text(encoding="utf-8"))
        return [_example_from_mapping(item) for item in data]
    if source.suffix.lower() == ".csv":
        with source.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if "question_id" in (reader.fieldnames or []):
                validate_eval_csv_schema(reader.fieldnames or [])
            return [_example_from_mapping(row) for row in reader]
    raise ValueError("Evaluation examples must be CSV or JSON.")


def validate_eval_csv_schema(fieldnames: list[str]) -> None:
    missing = sorted(REQUIRED_EVAL_COLUMNS.difference(fieldnames))
    if missing:
        raise ValueError(f"Evaluation CSV is missing required columns: {', '.join(missing)}")


def save_evaluation_outputs(
    result: EvaluationResult,
    csv_path: str | Path = "outputs/eval_results.csv",
    report_path: str | Path = "outputs/evaluation_report.md",
    document_names: list[str] | None = None,
    dataset_description: str = "Synthetic banking and enterprise operations benchmark.",
) -> None:
    csv_target = Path(csv_path)
    report_target = Path(report_path)
    csv_target.parent.mkdir(parents=True, exist_ok=True)
    report_target.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "question_id",
        "question",
        "question_type",
        "difficulty",
        "answerable",
        "expected_doc",
        "expected_page",
        "expected_keywords",
        "retrieved_chunk_ids",
        "retrieved_docs",
        "top_score",
        "hit_at_3",
        "hit_at_5",
        "reciprocal_rank",
        "returned_no_answer",
        "citation_present",
        "latency",
        "answer_preview",
        "failure_reason",
    ]
    with csv_target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for item in result.details:
            writer.writerow({key: _format_csv_value(item.get(key)) for key in fieldnames})

    report_target.write_text(
        _render_report(result, document_names=document_names or [], dataset_description=dataset_description),
        encoding="utf-8",
    )


def determine_failure_reason(example: EvaluationExample, item: dict) -> str:
    if not example.answerable:
        return "none" if item["returned_no_answer"] else "should_have_refused_but_answered"

    if item["returned_no_answer"]:
        return "should_have_answered_but_returned_no_answer"

    expected_docs = set(example.expected_document_ids)
    retrieved_docs = set(item["retrieved_docs"])
    if expected_docs and not expected_docs.issubset(retrieved_docs):
        return "missed_expected_doc"

    if example.expected_page is not None and expected_docs:
        page_found = any(
            doc in expected_docs and page == example.expected_page
            for doc, page in zip(item["retrieved_docs"], item["retrieved_pages"])
        )
        if not page_found:
            return "missed_expected_page"

    if example.expected_keywords:
        answer_text = item["answer"].lower()
        if any(keyword.lower() not in answer_text for keyword in example.expected_keywords):
            return "missing_expected_keyword"

    if not item["citation_present"]:
        return "missing_citation"

    return "none"


def _mean_metric(items: list[dict], key: str) -> float:
    values = [float(item[key]) for item in items if item.get(key) is not None]
    return round(mean(values), 4) if values else 0.0


def _citation_rate(items: list[dict]) -> float:
    answered_items = [item for item in items if item["is_answered"]]
    if not answered_items:
        return 0.0
    return round(mean([float(item["citation_present"]) for item in answered_items]), 4)


def _group_metrics(items: list[dict], group_key: str) -> dict[str, dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        grouped[str(item.get(group_key) or "unknown")].append(item)
    return {name: _metrics_for_items(group_items) for name, group_items in sorted(grouped.items())}


def _metrics_for_items(items: list[dict]) -> dict:
    answerable_items = [item for item in items if item["answerable"]]
    unanswerable_items = [item for item in items if not item["answerable"]]
    return {
        "questions": len(items),
        "answerable": len(answerable_items),
        "unanswerable": len(unanswerable_items),
        "hit_at_3": _mean_metric(answerable_items, "hit_at_3") if answerable_items else None,
        "hit_at_5": _mean_metric(answerable_items, "hit_at_5") if answerable_items else None,
        "mrr": _mean_metric(answerable_items, "reciprocal_rank") if answerable_items else None,
        "no_answer_accuracy": _mean_metric(unanswerable_items, "no_answer_correct") if unanswerable_items else None,
        "citation_rate": _citation_rate(items),
        "average_latency_ms": round(mean([item["latency_ms"] for item in items]), 2) if items else 0.0,
    }


def _example_from_mapping(item: dict) -> EvaluationExample:
    return EvaluationExample(
        question=str(item["question"]),
        answerable=_to_bool(item.get("answerable", True)),
        question_id=str(item.get("question_id", "")).strip(),
        question_type=str(item.get("question_type", "fact_lookup")).strip() or "fact_lookup",
        difficulty=str(item.get("difficulty", "medium")).strip() or "medium",
        expected_chunk_ids=_split_ids(item.get("expected_chunk_ids", "")),
        expected_document_ids=_split_ids(item.get("expected_document_ids") or item.get("expected_doc", "")),
        expected_keywords=_split_ids(item.get("expected_keywords", "")),
        expected_page=_to_optional_int(item.get("expected_page")),
        expected_answer=item.get("expected_answer") or None,
        notes=str(item.get("notes", "")).strip(),
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


def _render_report(result: EvaluationResult, document_names: list[str] | None = None, dataset_description: str = "") -> str:
    details = result.details
    interpretation = _interpret_result(result)
    distribution = Counter(item["question_type"] for item in details)
    successes = [item for item in details if item["failure_reason"] == "none" and item["answerable"]][:5]
    failures = [item for item in details if item["failure_reason"] != "none"][:5]
    lines = [
        "# Enterprise Document RAG QA System - Evaluation Report",
        "",
        "## Dataset Description",
        "",
        "- Project name: Enterprise Document RAG QA System",
        f"- Description: {dataset_description}",
        f"- Number of evaluation questions: {result.total_examples}",
        f"- Answerable questions: {result.answerable_count}",
        f"- Unanswerable questions: {result.unanswerable_count}",
        "",
        "## Document List",
        "",
    ]
    if document_names:
        lines.extend([f"- {name}" for name in document_names])
    else:
        lines.append("- Not provided")

    lines.extend(["", "## Question Distribution", "", "| Question type | Count |", "| --- | ---: |"])
    for question_type, count in sorted(distribution.items()):
        lines.append(f"| {question_type} | {count} |")

    lines.extend(
        [
            "",
            "## Overall Metrics",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
            f"| Questions | {result.total_examples} |",
            f"| Answerable questions | {result.answerable_count} |",
            f"| Unanswerable questions | {result.unanswerable_count} |",
            f"| Hit@3 | {result.hit_at_3:.4f} |",
            f"| Hit@5 | {result.hit_at_5:.4f} |",
            f"| MRR | {result.mrr:.4f} |",
            f"| No-answer accuracy | {result.no_answer_accuracy:.4f} |",
            f"| Citation rate | {result.citation_rate:.4f} |",
            f"| Average latency (ms) | {result.average_latency_ms:.2f} |",
            "",
            "## Metrics By Question Type",
            "",
        ]
    )
    lines.extend(_render_metrics_table(result.metrics_by_question_type))
    lines.extend(["", "## Metrics By Difficulty", ""])
    lines.extend(_render_metrics_table(result.metrics_by_difficulty))
    lines.extend(["", "## Successful Examples", ""])
    if not successes:
        lines.append("No fully successful answerable examples were recorded.")
    for item in successes:
        lines.extend(
            [
                f"### {item['question_id']}: {item['question']}",
                "",
                f"- Type: {item['question_type']} / {item['difficulty']}",
                f"- Retrieved docs: {', '.join(item['retrieved_docs']) or 'None'}",
                f"- Answer preview: {item['answer_preview']}",
                "",
            ]
        )

    lines.extend(["", "## Failure Or Weak Examples", ""])
    if not failures:
        lines.append("No failures were recorded in this run.")
    for item in failures:
        lines.extend(
            [
                f"### {item['question_id']}: {item['question']}",
                "",
                f"- Failure reason: {item['failure_reason']}",
                f"- Type: {item['question_type']} / {item['difficulty']}",
                f"- Expected doc: {item['expected_doc'] or 'None'}",
                f"- Retrieved docs: {', '.join(item['retrieved_docs']) or 'None'}",
                f"- Answer preview: {item['answer_preview']}",
                "",
            ]
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            interpretation,
            "",
            "## Limitations",
            "",
            "- The benchmark documents are synthetic and intentionally small compared with a real bank knowledge base.",
            "- TXT files expose only page 1 in the current parser, so page-level evaluation is limited for this benchmark.",
            "- MockProvider gives deterministic local answers and validates system wiring, but it does not measure final OpenAI answer quality.",
            "- Hashing embeddings are used by the evaluation CLI to avoid external downloads; production semantic retrieval uses sentence-transformers when available.",
            "- Cross-document questions are challenging because this version uses simple context selection and no reranker.",
            "",
            "## Next Improvement Suggestions",
            "",
            "- Add PDF/DOCX versions of the benchmark documents to exercise real page metadata.",
            "- Add human-labeled relevant chunk IDs for stricter retrieval evaluation.",
            "- Evaluate once with OpenAI enabled and compare answer quality against MockProvider.",
            "- Add optional reranking only after the current baseline is well understood.",
        ]
    )
    return "\n".join(lines)


def _interpret_result(result: EvaluationResult) -> str:
    if result.total_examples == 0:
        return "No evaluation examples were provided."
    if result.hit_at_3 >= 0.8 and result.no_answer_accuracy >= 0.8 and result.citation_rate >= 0.9:
        return (
            "The benchmark indicates a strong local baseline for retrieval, citation behavior, and no-answer handling. "
            "Failures should still be reviewed because the benchmark is synthetic and uses MockProvider."
        )
    return (
        "The benchmark surfaced quality gaps. Review the failure examples and grouped metrics before presenting this "
        "configuration as a mature RAG system."
    )


def _render_metrics_table(metrics: dict[str, dict]) -> list[str]:
    lines = [
        "| Group | Questions | Answerable | Unanswerable | Hit@3 | Hit@5 | MRR | No-answer accuracy | Citation rate | Avg latency ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for group, values in metrics.items():
        lines.append(
            f"| {group} | {values['questions']} | {values['answerable']} | {values['unanswerable']} | "
            f"{_format_metric(values['hit_at_3'])} | {_format_metric(values['hit_at_5'])} | "
            f"{_format_metric(values['mrr'])} | {_format_metric(values['no_answer_accuracy'])} | "
            f"{_format_metric(values['citation_rate'])} | {_format_metric(values['average_latency_ms'], digits=2)} |"
        )
    return lines


def _format_metric(value: object, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def _preview(text: str, limit: int = 240) -> str:
    cleaned = " ".join(text.split())
    return cleaned[:limit]
