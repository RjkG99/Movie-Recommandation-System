# -*- coding: utf-8 -*-
''' This is movie recomdasation system
    @learning '''
    
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity 
import pandas as pd
import numpy as np
text = ["London Paris London","Paris Paris London"]
cv = CountVectorizer()
count_matrix = cv.fit_transform(text)
print (count_matrix.toarray())
similarity_scores = cosine_similarity (count_matrix)
print(similarity_scores) 
df = pd.read_csv("movie_dataset.csv")
df =df.iloc[ : ,0:24]
print(df.columns)
features = ['keywords','cast','genres','director']
for feature in features:
    df[feature] = df[feature].fillna(' ')
def combine_features(row):
    return row['keywords']+" "+row['cast']+" "+row['genres']+" "+row['director']
df["combine_features"] = df.apply(combine_features,axis=1)
print(df["combine_features"].head())
cv = CountVectorizer()
count_matrix= cv.fit_transform(df["combine_features"])
count_matrix.toarray()
cosine_similarity = cosine_similarity(count_matrix)