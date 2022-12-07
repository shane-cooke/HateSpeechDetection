import pandas as pd
import numpy as np
import re, string
import nltk
import pickle
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer
import time

df = pd.read_csv('/Users/shanecooke/Desktop/Official GitLab/CompleteData.csv')
df

def preProcessText(text):
    text = text.lower() 
    text = text.strip()  
    text = re.compile('<.*?>').sub('', text) 
    text = re.compile('[%s]' % re.escape(string.punctuation)).sub(' ', text)  
    text = re.sub('\s+', ' ', text)  
    text = re.sub(r'[^\w\s]', '', str(text).lower().strip())
    text = re.sub(r'\d',' ',text) 
    text = re.sub(r'\s+',' ',text) 
    return text

def stopwordRemoval(string):
    stop = [i for i in string.split() if i not in stopwords.words('english')]
    return ' '.join(stop)

wl = WordNetLemmatizer()

def tagMapping(tag):
    if tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
    
def lemmatization(string):
    words = nltk.pos_tag(word_tokenize(string))
    temp = [wl.lemmatize(tag[0], tagMapping(tag[1])) for idx, tag in enumerate(words)]
    return " ".join(temp)

def finalCleaning(string):
    return lemmatization(stopwordRemoval(preProcessText(string)))

df['clean_text'] = df['Comment'].apply(lambda x: finalCleaning(x))

sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')
sentence_embeddings = sbert_model.encode(df['clean_text'])
BERT_list = np.array(sentence_embeddings).tolist()
df['BERT'] = BERT_list

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.metrics import precision_score, recall_score
from statistics import mean
from xgboost import XGBClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

X, y = df.BERT.tolist(), df.Hateful

f1_0 = []
f1_1 = []
rec_0 = []
rec_1 = []
prec_0 = []
prec_1 = []
a = []

t0 = time.time()

for i in range(20):
    
    t2 = time.time()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
    classifier = RandomForestClassifier(n_estimators=1000)
    #classifier = DecisionTreeClassifier()
    #classifier = GaussianNB()
    #classifier = SVC(kernel="rbf")
    #classifier = AdaBoostClassifier()
    #classifier = GaussianProcessClassifier()
    #classifier = KNeighborsClassifier()
    #classifier = MLPClassifier(identity='relu')
    #classifier = XGBClassifier()
    #classifier = LinearDiscriminantAnalysis(solver='eigen')
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    
    prec = precision_score(y_test, y_pred, average=None)
    rec = recall_score(y_test, y_pred, average=None)
    f1score = f1_score(y_test, y_pred, average=None)
    accuracy = accuracy_score(y_test, y_pred)
    
    if((prec[0] != 0.0) & (prec[1] != 0)):
        prec_0.append(prec[0])
        prec_1.append(prec[1])
    if((rec[0] != 0.0) & (rec[1] != 0)):
        rec_0.append(rec[0])
        rec_1.append(rec[1])
    if((f1score[0] != 0.0) & (f1score[1] != 0)):
        f1_0.append(f1score[0])
        f1_1.append(f1score[1])
    a.append(accuracy)
    i = i + 1
    t3 = time.time()
    
print("Accuracy: ", round(mean(a), 6))
print("Precision(0): ", round(mean(prec_0), 2), "    Precision(1): ", round(mean(prec_1), 2))
print("Recall(0): ", round(mean(rec_0),2), "    Recall(1): ", round(mean(rec_1), 2))
print("F1 Score(0): ", round(mean(f1_0), 2), "    F1 Score(1): ", round(mean(f1_1), 2))

t1 = time.time()
total = t1-t0
total2 = t3-t2
print("\nTime Taken (01): ", round(total2, 4), " seconds.")
print("\nTime Taken (20): ", round(total, 4), " seconds.")