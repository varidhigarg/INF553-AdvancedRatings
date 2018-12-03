from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer


FILE_MODEL = "sentiment_model.joblib"
FILE_CV = "countvectorizer_model.joblib"



final_model = joblib.load(FILE_MODEL)

cv = joblib.load(FILE_CV)

def get_sentiment(text):
	x_input=[]
	x_val = []
	x_input = [y for y in text.split(".")]
	X_test = cv.transform(x_input)
	return final_model.predict(X_test)
	
	
if __name__ == "__main__":
	print(get_sentiment("I didn't like the food. The ambiance was not good"))

