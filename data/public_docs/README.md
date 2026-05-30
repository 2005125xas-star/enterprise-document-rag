# Public Demo Documents

This folder is reserved for official public documents used by the public real-document demo.

PDF files are not committed to the repository. They are local evaluation artifacts and are ignored by `.gitignore`.

Only use official public documents here. Do not use private coursework, VLE/Moodle/Learning Mall/Canvas files, internal university documents, or copyrighted materials without permission.

## Required Filenames

The public evaluation runner expects these filenames:

- `ucl_chapter_4_assessment_framework_2025_26.pdf`
- `ucl_chapter_1_student_recruitment_admissions_2025_26.pdf`
- `liverpool_code_of_practice_on_assessment_2025_26.pdf`
- `liverpool_academic_integrity_policy_2025_26.pdf`
- `liverpool_student_attendance_policy_2025_26.pdf`

## Download And Evaluate

Download the official public PDFs:

```bash
python scripts/download_public_docs.py
```

Then run the public evaluation:

```bash
python -m src.evaluation.run_public_eval
```

The downloader skips existing files unless `--force` is passed:

```bash
python scripts/download_public_docs.py --force
```
