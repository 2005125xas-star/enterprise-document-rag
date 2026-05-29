# Demo Transcript - Mock Mode

This transcript was generated with the local `MockProvider` and deterministic hashing embeddings. It demonstrates retrieval, citation behavior, and no-answer handling, but it should not be interpreted as a real LLM answer-quality benchmark.

## Example 1 - Employee Referral Bonus

**Question:** What cap applies to employee referral bonuses?

**Should answer or refuse:** Answer.

**Retrieved sources:**

- `bank_recruitment_policy.txt`, page 1, chunk `bank_recruitment_policy:p1:c4`, score `0.483`
- `bank_recruitment_policy.txt`, page 1, chunk `bank_recruitment_policy:p1:c1`, score `0.095`
- `data_security_policy.txt`, page 1, chunk `data_security_policy:p1:c1`, score `0.090`

**Answer:** Based on the uploaded documents, 5.3 Employee referral bonuses are capped at 2,000 USD and are paid after the new hire completes 180 days of employment. [1]

**Why the answer is grounded:** The top retrieved chunk directly states the referral bonus cap and timing condition. The answer cites source `[1]`.

## Example 2 - Restricted Data Encryption

**Question:** What encryption standard is required for Restricted data at rest?

**Should answer or refuse:** Answer.

**Retrieved sources:**

- `data_security_policy.txt`, page 1, chunk `data_security_policy:p1:c1`, score `0.597`
- `data_security_policy.txt`, page 1, chunk `data_security_policy:p1:c2`, score `0.383`
- `data_security_policy.txt`, page 1, chunk `data_security_policy:p1:c3`, score `0.334`

**Answer:** Based on the uploaded documents, Encryption and Access Control 2.1 Restricted data must be encrypted at rest using AES-256 or an approved equivalent. [1]

**Why the answer is grounded:** The retrieved security policy chunk contains the exact encryption standard, `AES-256`, and the answer includes a citation.

## Example 3 - Teller Cash Drawer Limit

**Question:** What is the teller cash drawer limit?

**Should answer or refuse:** Answer.

**Retrieved sources:**

- `branch_operations_manual.txt`, page 1, chunk `branch_operations_manual:p1:c1`, score `0.493`
- `branch_operations_manual.txt`, page 1, chunk `branch_operations_manual:p1:c2`, score `0.184`
- `retail_banking_product_guide.txt`, page 1, chunk `retail_banking_product_guide:p1:c4`, score `0.107`

**Answer:** Based on the uploaded documents, Cash Controls 2.1 The standard teller cash drawer limit is 10,000 USD. [1]

**Why the answer is grounded:** The top branch operations chunk states the teller cash drawer limit directly.

## Example 4 - Cross-Document Direct Deposit Comparison

**Question:** Compare the direct deposit amount that waives checking fees with the bonus campaign deposit total.

**Should answer or refuse:** Answer, but this is a weak mock-mode example.

**Retrieved sources:**

- `customer_marketing_campaign_guide.txt`, page 1, chunk `customer_marketing_campaign_guide:p1:c1`, score `0.510`
- `retail_banking_product_guide.txt`, page 1, chunk `retail_banking_product_guide:p1:c1`, score `0.392`
- `customer_marketing_campaign_guide.txt`, page 1, chunk `customer_marketing_campaign_guide:p1:c2`, score `0.349`

**Answer:** Based on the uploaded documents, Spring Direct Deposit Bonus 2026 2.1 The Spring Direct Deposit Bonus campaign runs from 2026-02-01 through 2026-04-30. [1]

**Why the answer is grounded:** Retrieval found both the campaign guide and retail product guide, which is the right evidence set. The deterministic MockProvider answer is incomplete because it quotes only the first matching sentence instead of synthesizing both direct deposit amounts. This is a useful limitation to discuss in interviews.

## Example 5 - Unsupported Cryptocurrency Product

**Question:** What is the bank's cryptocurrency custody product code?

**Should answer or refuse:** Refuse.

**Retrieved sources:**

- `retail_banking_product_guide.txt`, page 1, chunk `retail_banking_product_guide:p1:c1`, score `0.489`
- `bank_recruitment_policy.txt`, page 1, chunk `bank_recruitment_policy:p1:c1`, score `0.397`
- `customer_marketing_campaign_guide.txt`, page 1, chunk `customer_marketing_campaign_guide:p1:c2`, score `0.297`

**Answer:** I could not find enough evidence in the uploaded documents.

**Why the answer is grounded:** The retrieved documents mention banking products and campaigns, but none contains a cryptocurrency custody product code. The evidence gate correctly returns the no-answer response.

