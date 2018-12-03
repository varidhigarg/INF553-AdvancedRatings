import json
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF
from sklearn.externals import joblib
import codecs


def display_topics(H, W, feature_names, documents, no_top_words, no_top_documents):
    for topic_idx, topic in enumerate(H):
        print("Topic %d:" % (topic_idx))
        print (" ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))
        top_doc_indices = np.argsort( W[:,topic_idx] )[::-1][0:no_top_documents]


def get_topic_model(no_topics, data_filename, model_filename, vectorizer_filename):
    with codecs.open(data_filename) as f:
        head = [json.loads(next(f)) for x in range(100000)]

    documents = []
    for i in head:
        documents.extend(i['text'].split('.'))

    # tf-idf
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
    tfidf = tfidf_vectorizer.fit_transform(documents)
    tfidf_feature_names = tfidf_vectorizer.get_feature_names()

    # Run NMF
    nmf_model = NMF(n_components=no_topics, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(tfidf)
    nmf_W = nmf_model.transform(tfidf)
    nmf_H = nmf_model.components_

    # dump model
    joblib.dump(nmf_model, model_filename)

    # dump vectorizer
    joblib.dump(tfidf_vectorizer, vectorizer_filename)

	
if __name__=="__main__":
	get_topic_model(5,'processed_reviews.json','nmf_model.joblib','vectorizer.joblib')

