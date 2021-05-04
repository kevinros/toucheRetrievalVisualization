import processor
import os
import json
import pickle
import hnswlib

# initialize the pyserini search
def initializePyserini(path_to_corpus_dir, path_to_corpus_output, path_to_idx_output):
    """
    Formats the json files to match pyserini input, then builds the index

    :param path_to_corpus_dir: path to where all the json files are
    :type path_to_corpus dir: str

    :param path_to_corpus_output: path where the each pyserini-formatted corpus is saved
    :type path_to_corpus_output: str

    :param path_to_idx_output: path where the pyserini index is saved
    :type path_to_idx_output: str

    :rtype: None
    :returns: Nothing
    """
    # Format and rewrite each corpus
    for corpus_name in os.listdir(path_to_corpus_dir):
        corpus = json.load(open(path_to_corpus_dir + corpus_name, 'r'))
        documents = []
        for doc in corpus['arguments']:
            text = [x['text'] for x in doc['premises']]
            documents.append({'id': doc['id'], 'contents': " ".join(text)})
        with open(path_to_corpus_output + corpus_name, 'w') as f:
            json.dump(documents, f)

    # Run pyserini command to build index
    # I know I know, this is probably the worst way to do it, I should just call the pyserini index directly
    cs_default = "python3 -m pyserini.index -collection JsonCollection -generator DefaultLuceneDocumentGenerator -threads 1"
    cs_input = " -input " + path_to_corpus_output
    cs_output = " -index " + path_to_idx_output + " -storePositions -storeDocvectors -storeRaw"
    os.system(cs_default + cs_input + cs_output)
    

def initializeSemantic(path_to_corpus_dir, path_to_semantic_output, model):
    """
    Encodes all corpus text and saves it in a hnswlib index.
    Note that this encoding happens on a passage level, which is just
    an arbitrary segmentation of each document (every 200 or so words).
    The assumption is that if a single passage is very relevant to a query,
    then the entire document is very relevant as well.

    :param path_to_corpus_dir: path to where all the json files are
    :type path_to_corpus dir: str

    :param path_to_semantic_output: directory where to save the index data
    :type path_to_semantic_output: str

    :param model: SentenceTransformer model used to encode passages
    :type model: SentenceTransformer

    :rtype: None
    :returns: Nothing
    """
    
    # Lookup arrays for the output idx of knn search 
    idx_to_passageid = []
    idx_to_passage = []

    # Index parameters currently hardcoded, seem to work fine
    # If a different corpus is used, the max_elements probably should be changed
    embedding_size = 768
    index = hnswlib.Index(space = 'cosine', dim = embedding_size)
    index.init_index(max_elements = 700000, ef_construction = 300, M = 64)

    # Loop through each corpus, processing the text
    for corpus_name in os.listdir(path_to_corpus_dir):
        corpus = json.load(open(path_to_corpus_dir + corpus_name, 'r'))
        print('processing corpus', corpus_name)
        for doc in corpus['arguments']:
            docid = doc['id']
            text = " ".join([x['text'] for x in doc['premises']])
            sentences = processor.createSentences(text)
            passages = processor.createPassages(sentences)
            # Save both the processed passages, and which document they came from
            idx_to_passage += passages
            idx_to_passageid += [docid] * len(passages)
            #for i in range(len(passages)):
            #    idx_to_passageid.append(docid + '.' + str(i))

    # Now that all the passages are made, we can encode them and add them to the index
    # Encode in batches of 1000 to make things a bit faster
    for i in range(0, len(idx_to_passage), 1000):
        encoded_passages = model.encode(idx_to_passage[i:i+1000])
        index.add_items(encoded_passages)
        if i % 10000 == 0: print(i, 'documents encoded')

    # Finally, the index and lookup arrays are saved to the provided path
    index.save_index(path_to_semantic_output + 'passage.index')
    pickle.dump(idx_to_passageid, open(path_to_semantic_output + 'idx_to_passageid.p', 'wb'))
    pickle.dump(idx_to_passage, open(path_to_semantic_output + 'idx_to_passage.p', 'wb'))
            
