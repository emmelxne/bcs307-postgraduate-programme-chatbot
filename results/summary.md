# Evaluation summary

## dev set (24 questions)

| Method | Correct | Accuracy | Outcomes |
|---|---|---|---|
| tfidf | 24/24 | 100% | Correct 19, Correct (NF) 5 |
| bm25 | 23/24 | 96% | Correct 18, Correct (NF) 5, Irrelevant 1 |
| hybrid | 24/24 | 100% | Correct 19, Correct (NF) 5 |

- **tfidf** by category: Factual 5/5, Paraphrased 6/6, Retrieval 8/8, Unavailable 5/5
- **bm25** by category: Factual 5/5, Paraphrased 6/6, Retrieval 7/8, Unavailable 5/5
- **hybrid** by category: Factual 5/5, Paraphrased 6/6, Retrieval 8/8, Unavailable 5/5

## heldout set (15 questions)

| Method | Correct | Accuracy | Outcomes |
|---|---|---|---|
| tfidf | 10/15 | 67% | Correct 8, Correct (NF) 2, Irrelevant 2, Missed 2, Unsupported 1 |
| bm25 | 9/15 | 60% | Correct 7, Correct (NF) 2, Irrelevant 3, Missed 2, Unsupported 1 |
| hybrid | 10/15 | 67% | Correct 8, Correct (NF) 2, Irrelevant 2, Missed 2, Unsupported 1 |

- **tfidf** by category: Factual 3/3, Paraphrased 2/5, Retrieval 3/4, Unavailable 2/3
- **bm25** by category: Factual 2/3, Paraphrased 2/5, Retrieval 3/4, Unavailable 2/3
- **hybrid** by category: Factual 3/3, Paraphrased 2/5, Retrieval 3/4, Unavailable 2/3
