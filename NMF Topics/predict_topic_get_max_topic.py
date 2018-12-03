import numpy as np
import json
from sklearn.externals import joblib



def init(model_filename, vectorizer_filename):
	global model
	global tfidf_vectorizer
	model = joblib.load(model_filename)
	tfidf_vectorizer = joblib.load(vectorizer_filename)

	
def predict_topic(sentence):
	sentence_topics = model.transform(tfidf_vectorizer.transform([sentence]))
	# get top topic
	'''top_topic = 0
	top_topic_val = float('-inf')
	for topic, topic_val in enumerate(sentence_topics[0]):
		if (topic_val > top_topic_val) and (topic != 2):
			top_topic_val = topic_val
			top_topic = topic
	return top_topic'''
	return sentence_topics[0].argmax(axis=0)
	
	
def get_normal():
	topics = {}
	file = 'processed_reviews.json'
	with open(file) as f:
		head = [json.loads(next(f)) for x in range(10000)]
		
	documents = []
	for i in head:
		documents.extend(i['text'].split('.'))
	init('nmf_model.joblib', 'vectorizer.joblib')
	test_documents = documents[:10000]

	for document in test_documents:
		for sentence in document.split('.'):
			sentence = sentence.strip()
			if sentence:
				sentence_topics = model.transform(tfidf_vectorizer.transform([document]))
				top_topic = sentence_topics[0].argmax(axis=0)
				if top_topic in topics:
					topics[top_topic].append(sentence)
				else:
					topics[top_topic] = [sentence]
	topic_counts = {topic: len(sentences) for topic, sentences in topics.items()}
	print (topic_counts)

	# topics and sentences perecentages
	topic_percentages = {topic: (len(sentences) * 1.0 / len(test_documents) * 100) for topic, sentences in topics.items()}
	print (topic_percentages)

	
get_normal()