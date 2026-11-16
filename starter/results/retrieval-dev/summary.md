Retrieval · split `dev` · N = câu có đáp án

| Config | Method | top-k | Filter | Neighbours | Budget | Hit@1 | Evidence Hit@3 | MRR | Context Evidence Rate | Avg context chars | Median ms |
|---|---|---|---|---|---|---|---|---|---|---|---|
| R1_dense | dense | 3 | yes | no | 4000 | 7/20 (35%) | 9/20 (45%) | 0.392 | 9/20 (45%) | 2676 | 60.2 |
| R1b_dense_nofilter | dense | 3 | no | no | 4000 | 7/20 (35%) | 9/20 (45%) | 0.392 | 9/20 (45%) | 2652 | 65.0 |
| R0_tfidf | tfidf | 3 | yes | no | 4000 | 6/20 (30%) | 6/20 (30%) | 0.3 | 6/20 (30%) | 2685 | 22.4 |
| R2_bm25 | bm25 | 3 | yes | no | 4000 | 2/20 (10%) | 10/20 (50%) | 0.267 | 10/20 (50%) | 2649 | 7.2 |
| R3_hybrid | hybrid | 3 | yes | no | 4000 | 5/20 (25%) | 10/20 (50%) | 0.35 | 10/20 (50%) | 2682 | 91.8 |
| R4_hybrid_rerank | hybrid_rerank | 3 | yes | no | 4000 | 8/20 (40%) | 12/20 (60%) | 0.492 | 12/20 (60%) | 2616 | 3965.1 |
| R5_hybrid_rerank_nb | hybrid_rerank | 3 | yes | yes | 8000 | 8/20 (40%) | 12/20 (60%) | 0.492 | 17/20 (85%) | 6815 | 3917.7 |
| R6_dense_k5 | dense | 5 | yes | no | 6000 | 7/20 (35%) | 9/20 (45%) | 0.434 | 13/20 (65%) | 4522 | 83.2 |
