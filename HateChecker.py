import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import tensorflow_hub as hub
import re, string
import nltk
import pickle
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
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
import streamlit as st

st.title("Hate Speech Detection")
st.text("Testing against 3000 posts and comments from Reddit, Twitter and 4Chan!")

df = pd.read_pickle("/Users/shanecooke/Desktop/FinalDatabase.pkl")

X, y = df.USE.tolist(), df.Hateful.tolist()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

classifier1 = RandomForestClassifier(n_estimators=1000, random_state=0)
classifier1.fit(X_train, y_train)

classifier2 = DecisionTreeClassifier(max_depth = 10, random_state=0)
classifier2.fit(X_train, y_train)

classifier3 = GaussianNB()
classifier3.fit(X_train, y_train)

classifier4 = SVC(kernel="rbf")
classifier4.fit(X_train, y_train)

classifier5 = AdaBoostClassifier()
classifier5.fit(X_train, y_train)

classifier6 = GaussianProcessClassifier()
classifier6.fit(X_train, y_train)

classifier7 = KNeighborsClassifier()
classifier7.fit(X_train, y_train)

classifier8 = MLPClassifier(alpha=1, max_iter=1000)
classifier8.fit(X_train, y_train)

classifier9 = XGBClassifier()
classifier9.fit(X_train, y_train)

classifier10 = LinearDiscriminantAnalysis()
classifier10.fit(X_train, y_train)

X, y = df.BERT.tolist(), df.Hateful.tolist()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

classifier11 = RandomForestClassifier(n_estimators=1000, random_state=0)
classifier11.fit(X_train, y_train)

classifier12 = DecisionTreeClassifier(max_depth = 10, random_state=0)
classifier12.fit(X_train, y_train)

classifier13 = GaussianNB()
classifier13.fit(X_train, y_train)

classifier14 = SVC(kernel="rbf")
classifier14.fit(X_train, y_train)

classifier15 = AdaBoostClassifier()
classifier15.fit(X_train, y_train)

classifier16 = GaussianProcessClassifier()
classifier16.fit(X_train, y_train)

classifier17 = KNeighborsClassifier()
classifier17.fit(X_train, y_train)

classifier18 = MLPClassifier(alpha=1, max_iter=1000)
classifier18.fit(X_train, y_train)

classifier19 = XGBClassifier()
classifier19.fit(X_train, y_train)

classifier20 = LinearDiscriminantAnalysis()
classifier20.fit(X_train, y_train)

embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')

df_results = pd.DataFrame()
model_list = ['Random Forest', 'Decision Tree', 'Naive Bayes', 'SVC', 'AdaBoost', 'Gaussian Process', 'K Neighbours', 'Multi-Layer Perceptron', 'XGBoost', 'Linear Discrimination', 'Random Forest', 'Decision Tree', 'Naive Bayes', 'SVC', 'AdaBoost', 'Gaussian Process', 'K Neighbours', 'Multi-Layer Perceptron', 'XGBoost', 'Linear Discrimination']
df_results['Embedding'] = ['USE', 'USE', 'USE', 'USE', 'USE', 'USE', 'USE', 'USE', 'USE', 'USE', 'BERT', 'BERT', 'BERT', 'BERT', 'BERT', 'BERT', 'BERT', 'BERT', 'BERT', 'BERT'] 
df_results['Model'] = model_list
df_results['Prediction'] = ['?'] * 20

def hate_speech_detection_copy():
    
    predict_list = []
    user = st.text_area("Enter any Post or Comment: ")
    if len(user) < 1:
        print("  ")
    else:
        sample = user
        data = embed([sample])
        data2 = sbert_model.encode([sample])
        a = classifier1.predict(data)
        b = classifier2.predict(data)
        c = classifier3.predict(data)
        d = classifier4.predict(data)
        e = classifier5.predict(data)
        f = classifier6.predict(data)
        g = classifier7.predict(data)
        h = classifier8.predict(data)
        i = classifier9.predict(data)
        j = classifier10.predict(data)
        k = classifier11.predict(data2)
        l = classifier12.predict(data2)
        m = classifier13.predict(data2)
        n = classifier14.predict(data2)
        o = classifier15.predict(data2)
        p = classifier16.predict(data2)
        q = classifier17.predict(data2)
        r = classifier18.predict(data2)
        s = classifier19.predict(data2)
        t = classifier20.predict(data2)
        
        predict_list.append(a[0].astype(int))
        predict_list.append(b[0].astype(int))
        predict_list.append(c[0].astype(int))
        predict_list.append(d[0].astype(int))
        predict_list.append(e[0].astype(int))
        predict_list.append(f[0].astype(int))
        predict_list.append(g[0].astype(int))
        predict_list.append(h[0].astype(int))
        predict_list.append(i[0].astype(int))
        predict_list.append(j[0].astype(int))
        predict_list.append(k[0].astype(int))
        predict_list.append(l[0].astype(int))
        predict_list.append(m[0].astype(int))
        predict_list.append(n[0].astype(int))
        predict_list.append(o[0].astype(int))
        predict_list.append(p[0].astype(int))
        predict_list.append(q[0].astype(int))
        predict_list.append(r[0].astype(int))
        predict_list.append(s[0].astype(int))
        predict_list.append(t[0].astype(int))
        
        average = np.mean(predict_list)
        
        for i in range(len(predict_list)):
            if(predict_list[i] == 1): 
                df_results['Prediction'][i] = 'Hateful'
            else:
                df_results['Prediction'][i] = 'Not Hateful'
        
        if(average >= 0.5):
            average = "{0:.1%}".format(average)
            st.header("This post was found to be Hateful.")
            st.text("Confidence Level: ")
            st.text(average)
        else:
            average = "{0:.1%}".format(1 - average)
            st.header("\n\nThis post was not found to be Hateful.")
            st.text("Confidence Level: ")
            st.text(average)
        
        st.dataframe(df_results)
            
hate_speech_detection_copy()