# InsightForge AI — RAG Evaluation Report

**Date:** 2026-09-03 18:55:32
**Fixture Document:** `data\fixtures\insightforge_test.pdf` (15 pages, 15 chunks)

## Summary Metrics

| Metric | Score |
|---|---|
| Questions Evaluated | 8 |
| Hit Rate (Recall@5 >= 1) | **100.0%** (8/8) |
| Overall Precision@5 | **0.2000** |
| Average Top Hybrid Score | **0.8858** |

## Question Breakdown

| # | Question | Expected Pages | Retrieved Pages | Top Score | Precision | Hit |
|---|---|---|---|---|---|---|
| 1 | What is Artificial Intelligence? | [1] | [1, 8, 5, 2, 6] | 0.8200 | 0.20 | ✅ |
| 2 | What are the characteristics of Artificial Intelligence? | [1] | [14, 1, 2, 8, 11] | 0.8541 | 0.20 | ✅ |
| 3 | What are the types of Artificial Intelligence? | [2] | [2, 1, 8, 14, 11] | 0.8905 | 0.20 | ✅ |
| 4 | What is the history of Artificial Intelligence? | [5] | [5, 1, 8, 2, 11] | 0.8693 | 0.20 | ✅ |
| 5 | What is an intelligent system? | [6] | [6, 7, 1, 2, 8] | 0.9235 | 0.20 | ✅ |
| 6 | What are the applications of Artificial Intelligence? | [8] | [8, 1, 2, 14, 10] | 0.9107 | 0.20 | ✅ |
| 7 | What are the current trends in Artificial Intelligence? | [11] | [11, 2, 8, 14, 1] | 0.8918 | 0.20 | ✅ |
| 8 | What are the characteristics of problems in Artificial Intelligence? | [14] | [14, 1, 2, 8, 11] | 0.9261 | 0.20 | ✅ |
