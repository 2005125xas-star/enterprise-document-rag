from __future__ import annotations

from scripts.download_public_docs import PUBLIC_DOCUMENTS


def test_public_downloader_manifest_matches_expected_sources() -> None:
    expected = {
        "ucl_chapter_4_assessment_framework_2025_26.pdf": "https://www.ucl.ac.uk/study/sites/study/files/4a-assessment_regulations_for_taught_programmes_2025-26_clean.pdf",
        "ucl_chapter_1_student_recruitment_admissions_2025_26.pdf": "https://www.ucl.ac.uk/study/sites/study/files/chapter_1_student_recruitment_and_admissions_2025-26.pdf",
        "liverpool_code_of_practice_on_assessment_2025_26.pdf": "https://www.liverpool.ac.uk/media/livacuk/tqsd/code-of-practice-on-assessment/code_of_practice_on_assessment.pdf",
        "liverpool_academic_integrity_policy_2025_26.pdf": "https://www.liverpool.ac.uk/media/livacuk/tqsd/code-of-practice-on-assessment/appendix_L_cop_assess.pdf",
        "liverpool_student_attendance_policy_2025_26.pdf": "https://www.liverpool.ac.uk/media/livacuk/student-administration/sas/studentadministration/attendance/StudentAttendancePolicy-REVISEDJUL2025%2Cv1.3.pdf",
    }

    actual = {document.filename: document.url for document in PUBLIC_DOCUMENTS}

    assert actual == expected
    assert all(url.startswith("https://") for url in actual.values())
    assert all(filename.endswith(".pdf") for filename in actual)
