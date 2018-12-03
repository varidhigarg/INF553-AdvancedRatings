from itertools import islice

import numpy as np
import pandas as pd
import gensim
import nltk

from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.stem.porter import *

np.random.seed(2018)
nltk.download('wordnet')

def lemmatize_stem(text):
    stemmer = PorterStemmer()
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def preprocess(text):
    results = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS:
            results.append(lemmatize_stem(token))
    return results


if __name__ == "__main__":

    file = '../yelp_dataset/yelp_academic_dataset_review.json'
    with open(file) as f:
        head = [next(f) for x in range(100)]

    data = map(lambda x: x.rstrip(), head)
    data_json_str = "[" + ','.join(data) + "]"

    reviews_df = pd.read_json(data_json_str)
    reviews_df = reviews_df[['text']]
    # print reviews_df.head(5)
    # print reviews_df[['text']]

    documents = reviews_df[['text']]
    documents['index'] = documents.index
    # print documents

    documents = documents[:-3]
    test_documents = documents[-3:]

    # preprocessing
    processed_docs = documents['text'].map(preprocess)

    # dictionary
    dictionary = gensim.corpora.Dictionary(processed_docs)

    # doc2bow
    bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

    # LDA
    lda = gensim.models.LdaModel(bow_corpus, num_topics=5, id2word=dictionary)
    for topic in lda.print_topics(-1):
        print topic

    # predict
    test_documents = test_documents['text']
    for document in test_documents:
        print '*' * 50
        print document
        for sentence in document.split('.'):
            sentence = sentence.strip()
            if sentence:
                print sentence
                bow_vector = dictionary.doc2bow(preprocess(sentence))
                for topic in sorted(lda[bow_vector], key=lambda tup: -1*tup[1]):
                    print topic
