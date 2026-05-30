# Enterprise Document RAG QA System - Evaluation Report

## Dataset Description

- Project name: Enterprise Document RAG QA System
- Description: Synthetic banking and enterprise operations benchmark with policy, product, risk, security, branch operations, and marketing documents.
- Number of evaluation questions: 46
- Answerable questions: 41
- Unanswerable questions: 5

## Document List

- bank_recruitment_policy.txt
- branch_operations_manual.txt
- credit_risk_policy_summary.txt
- customer_marketing_campaign_guide.txt
- data_security_policy.txt
- retail_banking_product_guide.txt

## Question Distribution

| Question type | Count |
| --- | ---: |
| cross_document | 6 |
| fact_lookup | 4 |
| no_answer | 5 |
| numeric_threshold | 14 |
| policy_lookup | 6 |
| product_lookup | 4 |
| role_responsibility | 7 |

## Overall Metrics

| Metric | Value |
| --- | ---: |
| Questions | 46 |
| Answerable questions | 41 |
| Unanswerable questions | 5 |
| Hit@3 | 1.0000 |
| Hit@5 | 1.0000 |
| MRR | 0.9878 |
| No-answer accuracy | 1.0000 |
| Citation rate | 1.0000 |
| Average latency (ms) | 1.36 |

## Metrics By Question Type

| Group | Questions | Answerable | Unanswerable | Hit@3 | Hit@5 | MRR | No-answer accuracy | Citation rate | Avg latency ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cross_document | 6 | 6 | 0 | 1.0000 | 1.0000 | 0.9167 | n/a | 1.0000 | 1.36 |
| fact_lookup | 4 | 4 | 0 | 1.0000 | 1.0000 | 1.0000 | n/a | 1.0000 | 1.41 |
| no_answer | 5 | 0 | 5 | n/a | n/a | n/a | 1.0000 | 0.0000 | 1.33 |
| numeric_threshold | 14 | 14 | 0 | 1.0000 | 1.0000 | 1.0000 | n/a | 1.0000 | 1.35 |
| policy_lookup | 6 | 6 | 0 | 1.0000 | 1.0000 | 1.0000 | n/a | 1.0000 | 1.32 |
| product_lookup | 4 | 4 | 0 | 1.0000 | 1.0000 | 1.0000 | n/a | 1.0000 | 1.40 |
| role_responsibility | 7 | 7 | 0 | 1.0000 | 1.0000 | 1.0000 | n/a | 1.0000 | 1.36 |

## Metrics By Difficulty

| Group | Questions | Answerable | Unanswerable | Hit@3 | Hit@5 | MRR | No-answer accuracy | Citation rate | Avg latency ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| easy | 26 | 26 | 0 | 1.0000 | 1.0000 | 1.0000 | n/a | 1.0000 | 1.36 |
| hard | 6 | 6 | 0 | 1.0000 | 1.0000 | 0.9167 | n/a | 1.0000 | 1.36 |
| medium | 14 | 9 | 5 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.35 |

## Successful Examples

### Q001: What system must hiring managers use before contacting candidates?

- Type: fact_lookup / easy
- Retrieved docs: bank_recruitment_policy, bank_recruitment_policy, bank_recruitment_policy, data_security_policy, bank_recruitment_policy
- Answer preview: Based on the uploaded documents, 1.3 Hiring managers must use the WorkPath requisition system for every role before contacting candidates. [1]

### Q003: Who validates the job family and interview plan for a requisition?

- Type: role_responsibility / easy
- Retrieved docs: bank_recruitment_policy, bank_recruitment_policy, bank_recruitment_policy, data_security_policy, credit_risk_policy_summary
- Answer preview: Based on the uploaded documents, 2.2 Talent Acquisition validates the job family, salary band, and interview plan within three business days. [1]

### Q004: What salary range applies to Branch Advisor B2 roles?

- Type: numeric_threshold / medium
- Retrieved docs: bank_recruitment_policy, bank_recruitment_policy, bank_recruitment_policy, bank_recruitment_policy, branch_operations_manual
- Answer preview: Based on the uploaded documents, 3.2 Branch Advisor B2 roles use a salary range of 52,000 to 68,000 USD. [1]

### Q005: How long is probation for risk and compliance roles?

- Type: policy_lookup / easy
- Retrieved docs: bank_recruitment_policy, bank_recruitment_policy, bank_recruitment_policy, credit_risk_policy_summary, bank_recruitment_policy
- Answer preview: Based on the uploaded documents, 5.2 The probationary period is 120 calendar days for risk, compliance, and model governance roles. [1]

### Q006: What cap applies to employee referral bonuses?

- Type: numeric_threshold / easy
- Retrieved docs: bank_recruitment_policy, bank_recruitment_policy, data_security_policy, credit_risk_policy_summary
- Answer preview: Based on the uploaded documents, 5.3 Employee referral bonuses are capped at 2,000 USD and are paid after the new hire completes 180 days of employment. [1]


## Failure Or Weak Examples

### Q002: What is the effective date of the recruitment policy?

- Failure reason: missing_expected_keyword
- Type: fact_lookup / easy
- Expected doc: bank_recruitment_policy
- Retrieved docs: bank_recruitment_policy, data_security_policy, customer_marketing_campaign_guide, bank_recruitment_policy, credit_risk_policy_summary
- Answer preview: Based on the uploaded documents, Bank Recruitment and Workforce Policy Synthetic evaluation document. [1]

### Q009: What direct deposit amount waives the Everyday Checking monthly fee?

- Failure reason: missing_expected_keyword
- Type: numeric_threshold / easy
- Expected doc: retail_banking_product_guide
- Retrieved docs: retail_banking_product_guide, retail_banking_product_guide, customer_marketing_campaign_guide, retail_banking_product_guide, customer_marketing_campaign_guide
- Answer preview: Based on the uploaded documents, Everyday Checking RBC-CHK-100 2.1 Everyday Checking requires a 25 USD minimum opening deposit. [1]

### Q028: When must night deposit bags be processed?

- Failure reason: missing_expected_keyword
- Type: policy_lookup / medium
- Expected doc: branch_operations_manual
- Retrieved docs: branch_operations_manual, customer_marketing_campaign_guide, branch_operations_manual, customer_marketing_campaign_guide, data_security_policy
- Answer preview: Based on the uploaded documents, 2.4 Night deposit bags must be processed by 10:00 a.m. [1]

### Q032: What is the campaign code for Spring Direct Deposit Bonus 2026?

- Failure reason: missing_expected_keyword
- Type: product_lookup / easy
- Expected doc: customer_marketing_campaign_guide
- Retrieved docs: customer_marketing_campaign_guide, customer_marketing_campaign_guide, retail_banking_product_guide, retail_banking_product_guide, customer_marketing_campaign_guide
- Answer preview: Based on the uploaded documents, Spring Direct Deposit Bonus 2026 2.1 The Spring Direct Deposit Bonus campaign runs from 2026-02-01 through 2026-04-30. [1]

### Q033: How much is the Spring Direct Deposit Bonus?

- Failure reason: missing_expected_keyword
- Type: numeric_threshold / easy
- Expected doc: customer_marketing_campaign_guide
- Retrieved docs: customer_marketing_campaign_guide, customer_marketing_campaign_guide, retail_banking_product_guide, customer_marketing_campaign_guide, retail_banking_product_guide
- Answer preview: Based on the uploaded documents, Spring Direct Deposit Bonus 2026 2.1 The Spring Direct Deposit Bonus campaign runs from 2026-02-01 through 2026-04-30. [1]


## Interpretation

The benchmark indicates a strong local baseline for retrieval, citation behavior, and no-answer handling. Failures should still be reviewed because the benchmark is synthetic and uses MockProvider.

## Limitations

- The benchmark documents are synthetic and intentionally small compared with a real bank knowledge base.
- TXT files expose only page 1 in the current parser, so page-level evaluation is limited for this benchmark.
- MockProvider gives deterministic local answers and validates system wiring, but it does not measure final OpenAI answer quality.
- Hashing embeddings are used by the evaluation CLI to avoid external downloads; production semantic retrieval uses sentence-transformers when available.
- Cross-document questions are challenging because this version uses simple context selection and no reranker.

## Next Improvement Suggestions

- Add PDF/DOCX versions of the benchmark documents to exercise real page metadata.
- Add human-labeled relevant chunk IDs for stricter retrieval evaluation.
- Evaluate once with OpenAI enabled and compare answer quality against MockProvider.
- Add optional reranking only after the current baseline is well understood.