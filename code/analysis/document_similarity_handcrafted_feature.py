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


# uselection_features_ChatGPT = '../../data/uselection_features_ChatGPT'
# uselection_features_GT = '../../data/uselection_features_GT'
#
# wiki_features_GT = '../../data/wiki_features_GT'
# wiki_features_ChatGPT = '../../data/wiki_features_ChatGPT'
# print(os.listdir(uselection_features_GT)[0])
# file_path = uselection_features_GT + '/uselection_features_GT.csv'
# print(file_path)
df = pd.DataFrame(r'C:\Users\amart\Desktop\IEEE_Journal\University of Houston\AI_Generated_Text_Classfier\data\uselection_features_GT\uselection_features_GT.csv')
print(df.columns)
