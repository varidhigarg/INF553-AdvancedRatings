import re
import nltk
import json
from nltk.stem import WordNetLemmatizer

REPLACE_NO_SPACE = re.compile("(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\()|(\))|(\[)|(\])")
REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)|(=)")

def remove_punc(line):
    review = REPLACE_NO_SPACE.sub("", line.lower())
    review = REPLACE_WITH_SPACE.sub(" ", review)
    return review

def lemmatize(line):
    lemmatizer = WordNetLemmatizer()
    line = remove_punc(line)
    return " ".join([lemmatizer.lemmatize(word, pos='v') for word in nltk.word_tokenize(line)])


def preprocess_text(review):
    lines = [x for x in nltk.sent_tokenize(review) if len(x)>1]
    return ". ".join([lemmatize(line) for line in lines])
	
	
def preprocess_review(review):
	x = json.loads(review)
	x["text"] = preprocess_text(x["text"])
	return json.dumps(x)
	
	
if __name__=="__main__":
	from joblib import Parallel, delayed
	import multiprocessing
	
	file = 'restaurant_review.json'
	print("reading data")
	with open(file,encoding="utf-8") as f:
		data = f.readlines()
	print("preprocessing")
	
	num_cores = multiprocessing.cpu_count()
	out = Parallel(n_jobs=num_cores)(delayed(preprocess_review)(i) for i in data)
		
	file = 'processed_reviews.json'
	print("writing")
	with open(file,"w",encoding="utf-8") as f:
		f.write("\n".join(out))