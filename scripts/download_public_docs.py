from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PUBLIC_DOCS_DIR = PROJECT_ROOT / "data" / "public_docs"
DEFAULT_TIMEOUT_SECONDS = 30


@dataclass(frozen=True)
class PublicDocument:
    url: str
    filename: str


PUBLIC_DOCUMENTS: tuple[PublicDocument, ...] = (
    PublicDocument(
        url="https://www.ucl.ac.uk/study/sites/study/files/4a-assessment_regulations_for_taught_programmes_2025-26_clean.pdf",
        filename="ucl_chapter_4_assessment_framework_2025_26.pdf",
    ),
    PublicDocument(
        url="https://www.ucl.ac.uk/study/sites/study/files/chapter_1_student_recruitment_and_admissions_2025-26.pdf",
        filename="ucl_chapter_1_student_recruitment_admissions_2025_26.pdf",
    ),
    PublicDocument(
        url="https://www.liverpool.ac.uk/media/livacuk/tqsd/code-of-practice-on-assessment/code_of_practice_on_assessment.pdf",
        filename="liverpool_code_of_practice_on_assessment_2025_26.pdf",
    ),
    PublicDocument(
        url="https://www.liverpool.ac.uk/media/livacuk/tqsd/code-of-practice-on-assessment/appendix_L_cop_assess.pdf",
        filename="liverpool_academic_integrity_policy_2025_26.pdf",
    ),
    PublicDocument(
        url="https://www.liverpool.ac.uk/media/livacuk/student-administration/sas/studentadministration/attendance/StudentAttendancePolicy-REVISEDJUL2025%2Cv1.3.pdf",
        filename="liverpool_student_attendance_policy_2025_26.pdf",
    ),
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download public university policy PDFs for local RAG evaluation.")
    parser.add_argument("--force", action="store_true", help="Re-download files even if they already exist.")
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"Per-file download timeout in seconds. Default: {DEFAULT_TIMEOUT_SECONDS}.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PUBLIC_DOCS_DIR,
        help="Directory where public PDFs should be saved.",
    )
    args = parser.parse_args(argv)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading public university policy PDFs for local evaluation.")
    print("Warning: downloaded PDFs are local evaluation artifacts and should not be committed.")
    print("Only use official public documents. Do not use private coursework, VLE/Moodle/Learning Mall/Canvas files, or internal documents.")
    print()

    failures = 0
    for document in PUBLIC_DOCUMENTS:
        ok = download_document(document, output_dir=output_dir, force=args.force, timeout=args.timeout)
        if not ok:
            failures += 1

    succeeded = len(PUBLIC_DOCUMENTS) - failures
    print()
    print(f"Download summary: {succeeded} succeeded, {failures} failed, {len(PUBLIC_DOCUMENTS)} total.")
    if failures:
        print("One or more downloads failed. Verify the URL, network connection, and official source page.")
        return 1
    return 0


def download_document(document: PublicDocument, output_dir: Path, force: bool = False, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> bool:
    target = output_dir / document.filename
    if target.exists() and not force:
        print(f"SKIP    {document.filename} already exists")
        return True

    temp_target = target.with_suffix(target.suffix + ".part")
    request = urllib.request.Request(
        document.url,
        headers={"User-Agent": "enterprise-document-rag-public-demo/1.0"},
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = getattr(response, "status", 200)
            if status >= 400:
                raise urllib.error.HTTPError(document.url, status, f"HTTP {status}", hdrs=response.headers, fp=None)
            with temp_target.open("wb") as handle:
                while True:
                    chunk = response.read(1024 * 64)
                    if not chunk:
                        break
                    handle.write(chunk)
        temp_target.replace(target)
        print(f"OK      {document.filename}")
        return True
    except (OSError, urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
        if temp_target.exists():
            temp_target.unlink()
        print(f"FAILED  {document.filename}: {exc}", file=sys.stderr)
        return False


if __name__ == "__main__":
    raise SystemExit(main())
