import processor

def semanticSearch(model, topics, index, idx_to_passageid, k=100):
    """
    Performs semantic similarity search over the corpus

    :param model: the SentenceTransformer encoder model
    :type model: SentenceTransformer

    :param topic: dict of the topic file
    :type topics: dict

    :param index: the hnswlib knn index
    :type index: hnswlib.Index

    :param idx_to_passageid: map from hnswlib index output to passage id
    :type idx_to_passageid: array

    :param k: number of neighbors to retrieve (default=100)
    :type k: int

    :rtype: dict
    :returns: dictionary where the keys are the topics and the values are sorted (docid, score) run lists

    """
    run = {}
    topic_nums = [topic for topic in topics]
    queries = [topics[topic]['title'] for topic in topics]
    encoded_queries = model.encode(queries)
    labels, distances = index.knn_query(encoded_queries, k=k)
    for i,topic in enumerate(topic_nums):
        run[topic] = []
        # considers highest passage match only for a document
        added_docids = []
        sim = [1-x for x in distances[i]]
        scored_run = zip(labels[i], sim)
        for i, (passageidx, dist) in enumerate(scored_run):
            docid = idx_to_passageid[passageidx].split('.')[0]
            if docid not in added_docids:
                run[topic].append((docid, sim))
                added_docids.append(docid)
    return run


def bm25Search(pyserini_searcher, topics, k1=3.2, b=0.15):
    """
    Performs BM25 search over the corpus

    :param pyserini_searcher: the pyserini SimpleSearcher instantiated on the corpora
    :type pyserini_searcher: pyserini SimpleSearcher

    :param topics: dict of the topic file
    :type topics: dict

    :param k1: BM25 parameter, optimized using last year's runs (default=3.2)
    :param b: BM25 parameter, optimized using last year's runs (default=0.15)

    :rtype: dict
    :returns: dictionary where the keys are the topics and the values are sorted (docid, score) run lists
    """
    
    pyserini_searcher.set_bm25(k1=k1, b=b)
    run = {}
    for i,topic in enumerate(topics):
        run[topic] = []
        query = topics[topic]['title']
        hits = pyserini_searcher.search(query, k=1000)
        # Sometimes, duplicate IDs get added, so this avoids adding duplicates
        # Can be significantly optimized
        for i in range(len(hits)):
            if hits[i].docid not in [x[0] for x in run[topic]]:
                run[topic].append((hits[i].docid, hits[i].score))
    return run
