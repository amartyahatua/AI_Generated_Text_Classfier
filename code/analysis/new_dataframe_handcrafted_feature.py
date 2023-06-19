import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import os
import csv



# uselection_features_ChatGPT = '../../data/uselection_features_ChatGPT'
# uselection_features_GT = '../../data/uselection_features_GT'
#
# wiki_features_GT = '../../data/wiki_features_GT'
# wiki_features_ChatGPT = '../../data/wiki_features_ChatGPT'
# print(os.listdir(uselection_features_GT)[0])
# file_path = uselection_features_GT + '/uselection_features_GT.csv'
# print(file_path)
path = r'C:\Users\amart\Desktop\IEEE_Journal\University of Houston\AI_Generated_Text_Classfier\data\features_ChatGPT\features_ChatGPT.csv'

features_ChatGPT = pd.DataFrame()
with open(path) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    column_name = None
    for row in csv_reader:
        if line_count == 0:
            print(f'Column names are {", ".join(row)}')
            column_name = row
            line_count += 1
        else:
            #print(f'\t{row[0]} works in the {row[1]} department, and was born in {row[2]}.')
            line_count += 1
            df_temp = pd.DataFrame([row], columns=column_name)
            features_ChatGPT = pd.concat([features_ChatGPT, df_temp], axis=0)
    print(f'Processed {line_count} lines.')
    print(features_ChatGPT.shape)
    features_ChatGPT.to_csv('features_ChatGPT.csv', index=False)