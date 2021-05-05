# Touché Task 1: Argument Retrieval for Controversial Questions
## Team Skeletor

### Five runs
1) **BM25 search**

k1=3.2, b=0.2, indexed via Pyserini using default settings.

2) **Semantic (dense) search**

Encoder: msmarco-distilbert-base-v3 from SentenceTransformers.<br />
Each document is segmented into BERT-sized passages, the passages are encoded and indexed in hnswlib.
Each topic title is encoded, knn search on the passages, take the max of all passage knn returned per document.
Each passage is about 200 words, and k=1000 similar passages are returned per topic title search.


3) **Interpolated BM25 and semantic search scores**

(1) + 0.7 * (2)


4) **Nearest-neighbor pseudo feedback via manifold approximation**

Using the run from (3), collect the passages from the top k highest-ranked documents (assumed to be relevant).
Then, do knn search using these passages across the entire corpus. Use the resulting ditances per passage to approximate
a local manifold for each passage. Aggregate the resulting edge existence probabilties per document.<br />
k = 3, knn = 50<br />
Note that this approach is modeled after UMAP manifold approximation: https://arxiv.org/pdf/1802.03426.pdf


5) **Nearest-neighbor pseudo feedback via manifold approximation, with cutoff**

Same approach as (4), however only rerank the top j documents from (3).<br />
k = 3, knn = 50, j = 10

### How to reproduce the runs
1) Clone the repository
2) Install the required python packages
3) In main.py, set the "initialize", "setup", and "evaluate" flags to True 
4) Run python3 main.py


### Description of files