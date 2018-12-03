import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import codecs



DATA_TO_READ = 100000
LABEL_POSITIVE = "P"
LABEL_NEGATIVE = "N"
FILE_MODEL = "sentiment_model.joblib"
FILE_CV = "countvectorizer_model.joblib"
FILE_INPUT = 'processed_reviews.json'

with codecs.open(FILE_INPUT, "r","utf-8") as f:
    data = [json.loads(next(f).strip()) for x in range(DATA_TO_READ)]

positive=[]
negative=[]

test=[]

for x in data:
    stars = x["stars"]
    if stars<2:
        negative.append(x["text"])
    if stars>4:
        positive.append(x["text"])
        

DATA_TO_TRAIN_EACH = min(len(negative),len(positive))


train_data = []
train_data.extend(positive[:DATA_TO_TRAIN_EACH])
train_data.extend(negative[:DATA_TO_TRAIN_EACH])

cv = CountVectorizer(binary=True)
cv.fit(train_data)
X = cv.transform(train_data)
target = [LABEL_POSITIVE if i < DATA_TO_TRAIN_EACH else LABEL_NEGATIVE for i in range(DATA_TO_TRAIN_EACH*2)]

#train model
final_model = LogisticRegression(C=0.05)
final_model.fit(X, target)

#save model
from sklearn.externals import joblib
joblib.dump(final_model, FILE_MODEL)
joblib.dump(cv, FILE_CV)