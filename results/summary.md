# Evaluation summary

Automatic keyword/status checks. Fill the `manual_verdict` column in the CSVs for the final reported score.

## dev set (24 questions)

| Method | Correct | Accuracy | Outcomes |
|---|---|---|---|
| tfidf | 24/24 | 100% | Correct 24 |
| bm25 | 23/24 | 96% | Correct 23, Irrelevant 1 |
| hybrid | 24/24 | 100% | Correct 24 |

- **tfidf** by category: Factual 5/5, Paraphrased 6/6, Retrieval 8/8, Unavailable 5/5
- **bm25** by category: Factual 5/5, Paraphrased 6/6, Retrieval 7/8, Unavailable 5/5
- **hybrid** by category: Factual 5/5, Paraphrased 6/6, Retrieval 8/8, Unavailable 5/5

## heldout set (15 questions)

| Method | Correct | Accuracy | Outcomes |
|---|---|---|---|
| tfidf | 12/15 | 80% | Correct 12, Missed 2, Unsupported 1 |
| bm25 | 11/15 | 73% | Correct 11, Irrelevant 1, Missed 2, Unsupported 1 |
| hybrid | 12/15 | 80% | Correct 12, Missed 2, Unsupported 1 |

- **tfidf** by category: Factual 3/3, Paraphrased 4/5, Retrieval 3/4, Unavailable 2/3
- **bm25** by category: Factual 2/3, Paraphrased 4/5, Retrieval 3/4, Unavailable 2/3
- **hybrid** by category: Factual 3/3, Paraphrased 4/5, Retrieval 3/4, Unavailable 2/3

## audit set (31 questions)

| Method | Correct | Accuracy | Outcomes |
|---|---|---|---|
| tfidf | 31/31 | 100% | Correct 31 |
| bm25 | 31/31 | 100% | Correct 31 |
| hybrid | 31/31 | 100% | Correct 31 |

- **tfidf** by category: Factual 3/3, Paraphrased 4/4, Out of scope 2/2, Unavailable 4/4, Incomplete 1/1, Meta 1/1, Switch 2/2, Related 2/2, Retrieval 9/9, Clarify 2/2, Verify 1/1
- **bm25** by category: Factual 3/3, Paraphrased 4/4, Out of scope 2/2, Unavailable 4/4, Incomplete 1/1, Meta 1/1, Switch 2/2, Related 2/2, Retrieval 9/9, Clarify 2/2, Verify 1/1
- **hybrid** by category: Factual 3/3, Paraphrased 4/4, Out of scope 2/2, Unavailable 4/4, Incomplete 1/1, Meta 1/1, Switch 2/2, Related 2/2, Retrieval 9/9, Clarify 2/2, Verify 1/1
