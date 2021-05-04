import initializer
import processor
import searcher
import reranker

import pickle
import os
import hnswlib
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder
from pyserini.search import SimpleSearcher

# corpus paths
path_to_corpus_dir = 'debates/'

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

# relevance scores path
path_to_qrels = 'metadata/touche2020-task1-relevance-args-me-corpus-version-2020-04-01-corrected.qrels'

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
        full_docid = pid#.split('.')[0]
        if full_docid not in docid_to_doc:
            docid_to_doc[full_docid] = []
        docid_to_doc[full_docid].append(idx_to_passage[i])
    
    # initialize the bm25 searcher
    pyserini_searcher = SimpleSearcher(path_to_idx_output)

    # create the run directory
    #os.mkdir(path_to_run_output)

    # load the topics
    topics = processor.load_topics(path_to_topics)
    
    # Run BM25
    bm25_run = searcher.bm25Search(pyserini_searcher, topics)
    processor.writeRelevanceFile(bm25_run, path_to_run_output + 'run.bm25', 'skeletor-bm25')
    os.system('../trec_eval/./trec_eval -m ndcg_cut.5 ' + path_to_qrels + ' ' +  path_to_run_output + 'run.bm25')

    # Run semantic search
    semantic_run = searcher.semanticSearch(semantic_model, topics, hnswlib_index, idx_to_passageid)
    processor.writeRelevanceFile(semantic_run, path_to_run_output + 'run.semantic', 'skeletor-semantic')
    os.system('../trec_eval/./trec_eval -m ndcg_cut.5 ' + path_to_qrels + ' ' +  path_to_run_output + 'run.semantic')


    # Interpolate BM25 and semantic with alpha=0.7
    interpolated_bm25_semantic = reranker.interpolate(path_to_run_output + 'run.bm25', path_to_run_output + 'run.semantic', 0.7)
    processor.writeRelevanceFile(interpolated_bm25_semantic, path_to_run_output + 'run.interpolated.bm25.semantic', 'skeletor-interpolated-bm25-0.7semantic')
    os.system('../trec_eval/./trec_eval -m ndcg_cut.5 ' + path_to_qrels + ' ' +  path_to_run_output + 'run.interpolated.bm25.semantic')


    #exit()
    
    # Cross encoder reranking
    #ce_bm25_run = reranker.crossEncode(path_to_run_output + 'run.inter.bm25.semantic', cross_encoder_model, topics, docid_to_doc, topk=10)
    #processor.writeRelevanceFile(ce_bm25_run, path_to_run_output + 'run.bm25.semantic.ce', 'bm25-0.7semantic-ce')
    #os.system('../trec_eval/./trec_eval -m ndcg_cut.5 ' + path_to_qrels + ' ' +  path_to_run_output + 'run.bm25.semantic.ce')
    #exit()
    
    # NN rerank
    
    nn_bm25_run = reranker.nn_pf(path_to_run_output + 'run.interpolated.bm25.semantic', semantic_model, topics, hnswlib_index, idx_to_passageid, docid_to_doc)
    processor.writeRelevanceFile(nn_bm25_run, path_to_run_output + 'run.interpolated.bm25.semantic.nn', 'nn')
    os.system('../trec_eval/./trec_eval -m ndcg_cut.5 ' + path_to_qrels + ' ' +  path_to_run_output + 'run.interpolated.bm25.semantic.nn')
    
    
    # Interpolate
    for alpha in [0.1, 0.3, 0.5, 0.7]:
        print(alpha)
        inter_run = reranker.interpolate(path_to_run_output + 'run.interpolated.bm25.semantic', path_to_run_output + 'run.interpolated.bm25.semantic.nn', alpha)
        processor.writeRelevanceFile(inter_run, path_to_run_output + 'run.inter.nn', 'interpolated.nn')
        os.system('../trec_eval/./trec_eval -m ndcg_cut.5 ' + path_to_qrels + ' ' +  path_to_run_output + 'run.inter.nn')
