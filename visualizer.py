import json
import math
from pyserini.search import SimpleSearcher
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer

class Visualizer:

    def __init__(self, out_dir, pyserini_index_path):
        """
        Initializes the visualizer class
        """
        self.out_dir = out_dir

        self.searcher = SimpleSearcher(pyserini_index_path)
        self.searcher.set_bm25(k1=3.2, b=0.15)

        # Semantic encoder model
        self.encoder = SentenceTransformer('stsb-distilbert-base')

        # Stores the transcript stream
        self.transcript = []
        self.transcript_time = []
        
        # Stores historical calculations for future reference
        self.history = []

        # The number of transcript entries to group when searching
        self.lookback = 5
        # The number of documents to return per search
        self.knn = 100

    def addToTranscript(self, text, timestamp=None):
        """
        Appends a new text window to the transcript. 
        Note that the addition is treated as a single temporal object for weighting

        :param text: string of transcript text (n most recent words)
        :type text: str

        :rtype: None
        :returns: None, but updates transcript with the new text
        """
        self.transcript.append(text)
        if timestamp == None:
            self.transcript_time.append(len(self.transcript_time))
        else:
            self.transcript_time.append(timestamp)
        run = self.weightedWindowSearch(lookback=self.lookback, k=self.knn, weight='uniform')
        run_metrics = {'dcg': 0, 'docs': {}}
        run_metrics['docs'] = {doc[0]: {'counter': 0, 'position': i, 'position_change': 0, 'score': doc[1], 'score_change': doc[1]} for i,doc in enumerate(run)}
        if len(self.history) > 0:
            run_metrics = self.compareRuns(self.history[-1], run_metrics)
        self.history.append(run_metrics)
        
    def weightedWindowSearch(self, lookback=1, k=1, weight=None):
        # Collect the transcript windows based on the lookback
        search_windows = self.transcript[-1 * lookback:]
        # Calculate the weights for each window
        if weight == 'uniform':
            search_window_weights = [1] * lookback
        elif weight == 'discount':
            search_window_weights = [1 / (lookback-i) for i in range(0, lookback)]
        run = {}
        for window, weight in zip(search_windows, search_window_weights):
            hits = self.searcher.search(window, k=100)
            for hit in hits:
                if hit.docid not in run:
                    run[hit.docid] = 0
                run[hit.docid] += hit.score * weight
        sorted_run = sorted([(docid, run[docid]) for docid in run], reverse=True, key=lambda x: x[1])[:k]
        return sorted_run

    def compareRuns(self, prev_run, cur_run):
        """
        Compares the position of documents across two given runs.
        DCG, positional gain, score gain
        """
        for doc in cur_run['docs']:
            if doc in prev_run['docs']:
                cur_run['docs'][doc]['counter'] = prev_run['docs'][doc]['counter'] + 1
                cur_run['docs'][doc]['position_change'] = prev_run['docs'][doc]['position'] - cur_run['docs'][doc]['position']
                cur_run['docs'][doc]['score_change'] = cur_run['docs'][doc]['score'] - prev_run['docs'][doc]['score']
                cur_run['dcg'] += 1 / math.log(cur_run['docs'][doc]['position'] + 2)

        return cur_run

    def findFreqDocs(self, start_idx, end_idx, topn=20):
        """
        Gets the topn most frequent docs over a given window
        """
        doc_counts = {}
        for run in self.history[start_idx:end_idx]:
            for doc in run['docs']:
                if doc not in doc_counts:
                    doc_counts[doc] = 0
                doc_counts[doc] += 1
        sorted_doc_counts = sorted([(doc, doc_counts[doc]) for doc in doc_counts], reverse=True, key=lambda x: x[1])
        return [doc[0] for doc in sorted_doc_counts][:topn]
        

    def getTopDocs(self, idx, topn=20):
        """
        Gets the topn docids for a given index
        """
        docs = [docid for docid in self.history[idx]['docs'] if self.history[idx]['docs'][docid]['position'] < topn]
        return docs


    def plotDocRankings(self, start_idx, end_idx, docs):
        """
        Plots the positions of the docis from the start_idx to the end_idx 
        """
        docs = {doc: [] for doc in docs}
        for i in range(start_idx, end_idx+1):
            for doc in docs:
                if doc in self.history[i]['docs']:
                    docs[doc].append(self.history[i]['docs'][doc]['position'])
                else:
                    docs[doc].append(self.knn)

        
        x = self.transcript_time[start_idx:end_idx+1]
        
        fig=plt.figure(figsize=(int((end_idx-start_idx) / 5),10))
        for doc in docs:
            plt.plot(x, docs[doc], label=doc)
        plt.xticks(rotation=45)

        ax = plt.gca()

        for index, label in enumerate(ax.xaxis.get_ticklabels()):
            if index % 4 != 0:
                label.set_visible(False)
        ax.invert_yaxis()

        plt.legend()

        

        plt.savefig(self.out_dir + 'docrankingsplot_' + str(start_idx) + '_' + str(end_idx+1) + '.png')
        
                    

        return docs

    def findDocsByKeyword(self, keywords_list, topn=5):
        """
        For each grouping of words in keyword_list, returns the topn doc ids
        
        :param keywords_list: independent strings of keywords to search
        :type keywords_list: list of strings
        
        :param topn: the number of documents to return for each keyword grouping
        :type topn: int

        :rtype: list of (keywords, list of doc ids) tuples
        :returns: pairs of keywords,document ids
        """
        docs = []
        for keywords in keywords_list:
            hits = self.searcher.search(keywords, k=topn)
            doc_ids = [hit.docid for hit in hits]
            docs.append((keywords, doc_ids))
        return docs
        
    
    def caterpillarEncode(self, start_idx, end_idx, docs, window=10, stride=1):
        """
        Splits sentences of docs, semantically embeds them
        Caterpillar embeds the transcript between the given indices
        Formats output to be used with projector.tensorflow.org
        """
        # get the document text
        doc_text = []
        doc_map = []
        for doc in docs:
            text = json.loads(self.searcher.doc(doc).raw())['contents'].split('. ') # crude preprocessing
            for sentence in text:
                if len(sentence.split()) > 10:
                    doc_text.append(sentence)
                    doc_map.append(doc)

        # get the transcript text:
        transcript_windows = []
        transcript_time = []
        for i in range(start_idx, end_idx - window):
            transcript_windows.append(" ".join(self.transcript[i:i+window]))
            transcript_time.append(self.transcript_time[i] + ' - ' + self.transcript_time[i+window])
            transcript_windows.append(" ".join(self.transcript[i:i+window+1]))
            transcript_time.append(self.transcript_time[i] + ' - ' + self.transcript_time[i+window+1])

            
        # merge and encode
        sentences = doc_text + transcript_windows
        labels = doc_map + transcript_time
        embeddings = self.encoder.encode(sentences) # assuming that whatever is being encoded isn't super large (10,000+ sentences)
        
        with open(self.out_dir + 'data.txt', 'w') as d, open(self.out_dir + '/metadata.txt', 'w') as m:
            m.write('Sentence\tLabel\tIndex\n')
            for i,(sentence, embedding) in enumerate(zip(sentences, embeddings)):
                if i < len(doc_map):
                    label = labels[i]
                    index = 'None'
                else:
                    label = 'Transcript'
                    index = labels[i]
                m.write(sentence + '\t' + label + '\t' + index + '\n')
                d.write( '\t'.join([str(x) for x in embedding.tolist()]) + "\n")
        
            
        

path_to_transcript = '../data/transcripts/debate.txt'

    
x = Visualizer('out/visualization/', 'out/pyserini/index/')
with open(path_to_transcript) as f:
    content = [line.strip() for line in f]
    text = []
    time = []
    for i in range(0, len(content), 2):
        time.append(content[i])
        text.append(content[i+1])

start_idx = time.index('110:53')
        
docs = []
for i,(segment, timestamp) in enumerate(zip(text[start_idx: start_idx + 100], time[start_idx: start_idx + 100])):
    print(i,segment)
    x.addToTranscript(segment, timestamp=timestamp)
    
    #docs = list(set(docs + x.getTopDocs(i)))
docs = x.findFreqDocs(0,99, topn=100)
#docs = x.findDocsByKeyword(["bible god creationism", "heavens astronomy stars"], topn=5)
print(docs)
#docs = [docid for doc in docs for docid in doc[1]]
with open(x.out_dir + 'docs.json', 'w') as f:
    for doc in docs:
        f.write(x.searcher.doc(doc).raw())
#rankings = x.plotDocRankings(0, 100, docs)
#rankings = x.plotDocRankings(100, 200, docs)
#rankings = x.plotDocRankings(200, 300, docs)
#rankings = x.plotDocRankings(300, 400, docs)
#rankings = x.plotDocRankings(400, 499, docs)
x.caterpillarEncode(0, 99, docs)

