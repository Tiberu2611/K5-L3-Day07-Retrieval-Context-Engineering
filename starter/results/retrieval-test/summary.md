Retrieval · split `test` · N = câu có đáp án

| Config | Method | top-k | Filter | Neighbours | Budget | Hit@1 | Evidence Hit@3 | MRR | Context Evidence Rate | Avg context chars | Median ms |
|---|---|---|---|---|---|---|---|---|---|---|---|
| R1_dense | dense | 3 | yes | no | 4000 | 2/6 (33%) | 2/6 (33%) | 0.333 | 2/6 (33%) | 2571 | 179.9 |
| R5_hybrid_rerank_nb | hybrid_rerank | 3 | yes | yes | 8000 | 3/6 (50%) | 4/6 (67%) | 0.556 | 4/6 (67%) | 7524 | 3262.5 |
