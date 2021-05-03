import initializer
import processor
import searcher

import pickle
import os
import hnswlib
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder
from pyserini.search import SimpleSearcher

# corpus paths
path_to_corpus_dir = 'rawDebatesSmall/'

# pyserini paths
path_to_corpus_output = 'out/pyserini/processedCorpus/'
path_to_idx_output = 'out/pyserini/index/'

# semantic paths
path_to_semantic_output = 'out/semantic/'

# run path
path_to_run_output = 'out/runs/'

# embedding models
semantic_model = SentenceTransformer('msmarco-distilbert-base-v3')
cross_encoder_model = CrossEncoder('cross-encoder/ms-marco-TinyBERT-L-6', max_length=512)

# topic path
path_to_topics = 'metadata/topics-task-1.xml' # note that these are the old titles

setup = False
evaluate = True

if setup:
    os.mkdir('out')
    os.mkdir('out/pyserini')
    os.mkdir(path_to_corpus_output)
    os.mkdir(path_to_idx_output)

    os.mkdir(path_to_semantic_output)
    
    initializer.initializePyserini(path_to_corpus_dir, path_to_corpus_output, path_to_idx_output)
    initializer.initializeSemantic(path_to_corpus_dir, path_to_semantic_output, semantic_model)

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
        if full_docid not in docid_to_doc:
            docid_to_doc[full_docid] = []
        docid_to_doc[full_docid].append(idx_to_passage[i])
    
    # initialize the bm25 searcher
    pyserini_searcher = SimpleSearcher(path_to_idx_output)

    # create the run directory
    os.mkdir(path_to_run_output)

    # load the topics
    topics = processor.load_topics(path_to_topics)
    
    # Run BM25
    bm25_run = searcher.bm25Search(pyserini_searcher, topics)
    processor.writeRelevanceFile(bm25_run, path_to_run_output + 'run.bm25', 'skeletor-bm25')

    # Run semantic search
    semantic_run = searcher.semanticSearch(semantic_model, topics, hnswlib_index, idx_to_passageid)
    processor.writeRelevanceFile(semantic_run, path_to_run_output + 'run.semantic', 'skeletor-semantic')
    

    
