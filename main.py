import initializer
import processor
import searcher

import os
import hnswlib
from sentence_transformers import SentenceTransformer
from pyserini.search import SimpleSearcher

# corpus paths
path_to_corpus_dir = ''

# pyserini paths
path_to_corpus_output = 'out/pyserini/processedCorpus/'
path_to_idx_output = 'out/pyserini/index/'

# semantic paths
path_to_semantic_output = 'out/semantic/'

# run path
path_to_run_output = 'out/runs'

# embedding models
semantic_model = SentenceTransformer('msmarco-distilbert-base-v3')
cross_encoder_model = SentenceTransformer('cross-encoder/ms-marco-TinyBERT-L-6', max_length=512)

# topic path
path_to_topics = 'topics-task-1-only-titles.xml'

setup = True
evaluate = True

if setup:
    os.mkdir('out')
    os.mkdir('out/pyserini')
    os.mkdir(path_to_corpus_output)
    os.mkdir(path_to_idx_output)

    os.mkdir(path_to_semantic_output)
    
    initializer.initializePyserini(path_to_corpus_dir, path_to_corpus_output, path_to_idx_output)
    initializer.initializeSemantic(path_to_corpus_dir, path_to_semantic_output, model)

if evaluate:

    # load the semantic knn index + passage lookup files
    hnswlib_index = hnswlib.Index(space = 'cosine', dim=768)
    hnswlib_index.load_index(path_to_semantic_output + 'passage.index')
    hnswlib_index.set_ef(250)
    
    idx_to_passageid = pickle.load(open(path_to_semantic_output + 'idx_to_passageid.p', 'rb'))
    idx_to_passage = pickle.load(open(path_to_semantic_output + 'idx_to_passage.p', 'rb'))

    # construct reverse lookup for full document reranking
    docid_to_doc = {}
    for i,pid in enumerate(idx_to_passageid):
        full_docid = pid.split('.')[0]
        if docid not in docid_to_doc:
            docid_to_doc[docid] = []
        docid_to_doc[docid].append(idx_to_passage[i])
    
    # initialize the bm25 searcher
    pyserini_searcher = SimpleSearcher(path_to_idx_output)

    # create the run directory
    os.mkdir(path_to_run_output)

    # load the topics
    topics = processor.load_topics(path_to_topics)
    
    # Run BM25
    bm25_run = searcher.bm25Search(pyserini_searcher, topics)
    writeRelevanceFile(bm25_run, path_to_run_output + 'run.bm25', 'skeletor-bm25')

    # Run semantic search
    semantic_run = searcher.semanticSearch(model, topics, hnswlib_index, idx_to_passageid)
    writeRelevanceFile(semantic_run, path_to_run_output + 'run.semantic', 'skeletor-semantic')
    

    
