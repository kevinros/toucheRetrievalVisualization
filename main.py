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
#path_to_topics = 'metadata/topics-task-1.xml' # note that these are the old titles
path_to_topics = 'metadata/topics-task-1-only-titles.xml'



# relevance scores path
path_to_qrels = 'metadata/touche2020-task1-relevance-args-me-corpus-version-2020-04-01-corrected.qrels'

# flag to create the directories, create the Pyserini index, and embedd the corpus
initialize = False
# flag to load the indexes
setup = True
# flag to run the searches
evaluate = True
# flag to view some results (setup must be true)
view = True

if initialize:
    os.mkdir('out')
    os.mkdir('out/pyserini')
    os.mkdir(path_to_corpus_output)
    os.mkdir(path_to_idx_output)

    os.mkdir(path_to_semantic_output)
    
    initializer.initializePyserini(path_to_corpus_dir, path_to_corpus_output, path_to_idx_output)
    initializer.initializeSemantic(path_to_corpus_dir, path_to_semantic_output, semantic_model)

if setup:

    # load the semantic knn index + passage lookup files
    hnswlib_index = hnswlib.Index(space = 'cosine', dim=768)
    hnswlib_index.load_index(path_to_semantic_output + 'passage.index')
    hnswlib_index.set_ef(1100)
    
    idx_to_docid = pickle.load(open(path_to_semantic_output + 'idx_to_passageid.p', 'rb'))
    idx_to_passage = pickle.load(open(path_to_semantic_output + 'idx_to_passage.p', 'rb'))

    # construct reverse lookup for full document reranking
    docid_to_doc = {}
    for i,full_docid in enumerate(idx_to_docid):
        if full_docid not in docid_to_doc:
            docid_to_doc[full_docid] = []
        docid_to_doc[full_docid].append(idx_to_passage[i])
    
    # initialize the bm25 searcher
    pyserini_searcher = SimpleSearcher(path_to_idx_output)

    # create the run directory (uncomment if first time)
    #os.mkdir(path_to_run_output)

    # load the topics
    topics = processor.load_topics(path_to_topics)

if evaluate:
    
    
    # Run BM25
    bm25_run = searcher.bm25Search(pyserini_searcher, topics)
    processor.writeRelevanceFile(bm25_run, path_to_run_output + 'run.bm25', 'bm25')
    #os.system('../trec_eval/./trec_eval -m ndcg_cut.5 ' + path_to_qrels + ' ' +  path_to_run_output + 'run.bm25')

    # Run semantic search
    semantic_run = searcher.semanticSearch(semantic_model, topics, hnswlib_index, idx_to_docid)
    processor.writeRelevanceFile(semantic_run, path_to_run_output + 'run.semantic', 'semantic')
    #os.system('../trec_eval/./trec_eval -m ndcg_cut.5 ' + path_to_qrels + ' ' +  path_to_run_output + 'run.semantic')


    # Interpolate BM25 and semantic with alpha=0.7
    interpolated_bm25_semantic = reranker.interpolate(path_to_run_output + 'run.bm25', path_to_run_output + 'run.semantic', 0.7)
    processor.writeRelevanceFile(interpolated_bm25_semantic, path_to_run_output + 'run.bm25.semantic', 'bm25-0.7semantic')
    #os.system('../trec_eval/./trec_eval -m ndcg_cut.5 ' + path_to_qrels + ' ' +  path_to_run_output + 'run.bm25.semantic')
   

    # NN manifold rerank no cutoff
    manifold_run = reranker.nn_pf_manifold(path_to_run_output + 'run.bm25.semantic', semantic_model, topics, hnswlib_index, idx_to_docid, docid_to_doc)
    processor.writeRelevanceFile(manifold_run, path_to_run_output + 'run.bm25.semantic.manifold', 'manifold')
    #os.system('../trec_eval/./trec_eval -m ndcg_cut.5 ' + path_to_qrels + ' ' +  path_to_run_output + 'run.bm25.semantic.manifold')

    # NN manifold rerank with 10 cutoff
    manifold_run_c10 = reranker.nn_pf_manifold(path_to_run_output + 'run.bm25.semantic', semantic_model, topics, hnswlib_index, idx_to_docid, docid_to_doc, rerank_cutoff=10)
    processor.writeRelevanceFile(manifold_run_c10, path_to_run_output + 'run.bm25.semantic.manifold_c10', 'manifold-c10')
    #os.system('../trec_eval/./trec_eval -m ndcg_cut.5 ' + path_to_qrels + ' ' +  path_to_run_output + 'run.bm25.semantic.manifold_c10')

        
if view:
    #given a run and an topic, print the top 10 most relevant documents
    bm25_run_path = path_to_run_output + 'run.bm25'
    semantic_run_path = path_to_run_output + 'run.semantic'
    bm25_semantic_run_path = path_to_run_output + 'run.bm25.semantic'
    manifold_run_path = path_to_run_output + 'run.bm25.semantic.manifold'
    manifold_c10_run_path = path_to_run_output + 'run.bm25.semantic.manifold_c10'

    topic = '91'
    for path_to_run in [manifold_run_path]:
         run = reranker.loadRun(path_to_run)[topic]
         print(topics[topic]['title'])
         for docid,_ in run[:5]:
             print(docid_to_doc[docid])
