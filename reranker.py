def loadRun(path_to_run):
    """
    Loads run into dict, where key is topic and value is array of (docid, score) tuples

    :param path_to_run: the path to the run
    :type path_to_run: str

    :rtype: dict
    :returns: dict of run
    """
    run = {}
    with open(path_to_run, 'r') as f:
        for line in f:
            split_line = line.strip().split()
            topic = split_line[0]
            docid = split_line[2]
            score = float(split_line[4])
            if topic not in run:
                run[topic] = []
            run[topic].append((docid, score))
    return run

def crossEncode(path_to_run, cross_encoder, topics, docid_to_doc, topk=20):
    """
    Reranks topk documents using cross-encoder
    
    :param path_to_run: the path to the run
    :type path_to_run: str
   
    :param cross_encoder: the cross encoder model
    :type cross_encoder: sentence_transformer CrossEncoder

    :param topics: dict of topics
    :type: dict

    :param docid_to_doc: a map from the document id to the document passages
    :type docid_to_doc: array

    :param topk: number of documents to rerank
    :type topk: int

    :rtype: dict
    :returns: dict of reranked run
    """
    run = loadRun(path_to_run)
    for topic in run:
        query = topics[topic]['title']
        print(query)
        reranked_run = []
        for docid,score in run[topic][:topk]:
            # Each passage is cross-encoded, and the score for a document is the average of all passage scores
            doc_score = 0
            try:
                for passage in docid_to_doc[docid]:
                    doc_score = max(doc_score, cross_encoder.predict([(query, passage)])[0])

                reranked_run.append((docid, doc_score))
            except Exception as e:
                print(e)
        sorted_run = sorted(reranked_run, reverse=True, key=lambda x: x[1])
        run[topic] = sorted_run
    return run
        
# umap search
def nn_pf(path_to_run, model, topics, index, idx_to_docid, docid_to_doc, rel_docs=5, k=20):
    """
    Assumes the top rel_docs are relevant, then does a k-nn search for all passages in those documents,
    and aggregates the scores of the similar passages

    :param path_to_run: path to the run to rerank
    :type path_to_run: str

    :param model: the semantic encoder
    :type model: SentenceTransformer

    :param topics: dict of the topics
    :type topics: dict

    :param index: the hnswlib index for knn search
    :type index: 

    :param idx_to_docid: the mapping between the hnswlib index output and the docid
    :type idx_to_docid: array

    :param docid_to_doc: the mapping between docid and the text in the doc
    :type docid_to_doc: dict

    :param rel_docs: number of relevant docs to gather passages from
    :type rel_docs: int
    
    :param k: the number of nearest neighbors to return
    :type k: int

    :rtype: dict
    :returns: dict of reranked run
    """


    run = loadRun(path_to_run)
    for topic in run:
        passages = []
        for docid,_ in run[topic][:rel_docs]:
            passages += docid_to_doc[docid]


        encoded_passages = model.encode(passages)
        scores = {}
        labels, distances = index.knn_query(encoded_passages, k=k)
        for i in range(len(encoded_passages)):
            for docidx, dist in zip(labels[i], distances[i]):
                docid = idx_to_docid[docidx]
                if docid not in scores:
                    scores[docid] = 0
                scores[docid] += 1-dist
        sorted_scores = sorted([(docidx, scores[docidx]) for docidx in scores], reverse=True, key=lambda x: x[1])
        run[topic] = sorted_scores
    return run


# interpolate runs
def interpolate(path_to_run1, path_to_run2, alpha):
    """
    Given to runs, combines the scores by run1 + (run2 * alpha)

    :param path_to_run1: path to the first run
    :type path_to_run1: str

    :param path_to_run2: path to the second run:
    :type path_to_run2: str

    :param alpha: how much of the second run should be added to the first run
    :type alpha: float

    :rtype: dict
    :returns: dict of reranked run
    """
    run1 = loadRun(path_to_run1)
    run2 = loadRun(path_to_run2)
    interpolated_runs = {}
    for topic in run1:
        # make run into dict
        interpolated_topic_run = {doc_score[0]: doc_score[1] for doc_score in run1[topic]}  
        for doc_score in run2[topic]:
            if doc_score[0] not in interpolated_topic_run:
                interpolated_topic_run[doc_score[0]] = alpha * doc_score[1]
            else:
                interpolated_topic_run[doc_score[0]] += alpha * doc_score[1]

        interpolated_topic_run = sorted([(doc, interpolated_topic_run[doc]) for doc in interpolated_topic_run], reverse=True, key=lambda x:x[1])
            
        interpolated_runs[topic] = interpolated_topic_run
    return interpolated_runs

        
