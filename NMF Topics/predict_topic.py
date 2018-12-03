import numpy as np

from sklearn.externals import joblib



def init(model_filename, vectorizer_filename):
	global model
	global tfidf_vectorizer
	model = joblib.load(model_filename)
	tfidf_vectorizer = joblib.load(vectorizer_filename)

	
def predict_topic( sentence):
	sentence_topics = model.transform(tfidf_vectorizer.transform([sentence]))
	# get top topic
	top_topic = 0
	top_topic_val = float('-inf')
	for topic, topic_val in enumerate(sentence_topics[0]):
		if (topic_val > top_topic_val) and (topic != 4):
			top_topic_val = topic_val
			top_topic = topic
	return top_topic
	# return topics[0].argmax(axis=0)