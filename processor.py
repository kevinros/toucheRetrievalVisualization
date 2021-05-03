import re
import json
import xml.etree.ElementTree as ElementTree

def load_topics(path, onlyTitles=True):
    """
    Loads the topics
    
    :param path: path to topic text file
    :type path: str
    
    :param onlyTitles: true if xml file has only titles, false otherwise, default true
    :type onlyTitles: bool
    
    :rtype: dict
    :return: dict where keys are the topics and values are the titles,descriptions, etc.
    """

    topics = {}
    with open(path, 'r') as f:
        xml = f.read()

    root = ElementTree.fromstring(xml)
    for topic in root:
        topic_num = topic.find('number').text.strip()
        topics[topic_num] = {}
        topics[topic_num]['title'] = topic.find('title').text.strip()
        if not onlyTitles:
            topics[topic_num]['description'] = topic.find('description').text.strip()
            topics[topic_num]['narrative'] = topic.find('narrative').text.strip()

    return topics




def writeRelevanceFile(run, output_path, run_name):
    """
    Writes a run to a relevance file, in trec-style format

    :param run: dict where keys are topics and values are (docid, score) tuples
    :type run: dict

    :param output_path: name of the file to write the results
    :type output_path: str

    :param run_name: name of the run
    :type run_name: str

    :rtype: None
    :returns: Nothing
    """ 
    with open(output_path, 'w') as f:
        for topic in run:
            for i,doc in enumerate(run[topic]):
                score = str(doc[1])
                outstr = topic + " Q0 " + doc[0] + " " + str(i+1) + " " + score + ' ' + run_name + '\n'
                f.write(outstr)

def createPassages(list_of_sentences, max_passage_words=200):
    """
    Segments the list of sentences into roughly equal_size passages

    :param list_of_sentences: list of strings
    :type list_of_sentences: list of strings

    :param max_passage_words: upper approximate limit on number of words for each passage
                              limited due to BERT encoder
                              default = 200
    :type max_passage_words: int

    :rtype: list of strings
    :returns: list where each entry is a passages
    """

    passages = []
    current_passage = []
    for sentence in list_of_sentences:
        current_passage.append(sentence)
        if len(" ".join(current_passage).split()) > max_passage_words:
            passages.append(" ".join(current_passage))
            current_passage = []
    # get any remaining sentences
    if current_passage:
        passages.append(" ".join(current_passage))
    return passages




def createSentences(text):
    """
    Lightweight sentence tokenizer based on regular expressions

    :param text: text to be processed
    :type text: str

    :rtype: list of strings
    :returns: list of sentences
    """
    alphabets= "([A-Za-z])"
    prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
    suffixes = "(Inc|Ltd|Jr|Sr|Co)"
    starters = "(|Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
    acronyms = "([A-Za-z][.][A-Za-z][.](?:[A-Za-z][.])?)"
    websites = "[.](com|net|org|io|gov)"

    text = " " + text + "  "
    text = text.replace("\n"," ")

    text = re.sub("=*=", ". ", text)
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
    if "" in text: text = text.replace(".",".")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [re.sub('\[[^a-zA-Z]+\]', ' ', sentence) for sentence in sentences]
    sentences = [re.sub(' +', ' ', sentence).strip() for sentence in sentences]

    # remove all sentences that are shorter than 3 words or longer than 500 words
    sentences = [s for s in sentences if len(s.split()) > 5 and len(s.split()) < 500]
    return sentences
