# source:https://medium.com/@adriensieg/text-similarities-da019229c894

import pandas as pd
import numpy as np
from nltk.corpus import stopwords
#import nltk

#nltk.download('stopwords')
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

model = SentenceTransformer('bert-base-nli-mean-tokens')

us_election_qa = pd.read_csv('../../data/chatgpt_generated_us_election_2024_questions_answers_combine.csv')

for i in tqdm(range(us_election_qa.shape[0])):
    if i > 0:
        break
    compare_embedd = []

    row_text = us_election_qa.iloc[i]['Text']
    row_text_embd = model.encode(row_text)

    row_ans1 = us_election_qa.iloc[i]['Answer 1']
    row_ans1_embd = model.encode(row_ans1)
    compare_embedd.append(row_ans1_embd)

    row_ans2 = us_election_qa.iloc[i]['Answer 2']
    row_ans2_embd = model.encode(row_ans2)
    compare_embedd.append(row_ans2_embd)

    row_ans3 = us_election_qa.iloc[i]['Answer 3']
    row_ans3_embd = model.encode(row_ans3)
    compare_embedd.append(row_ans3_embd)

    row_ans4 = us_election_qa.iloc[i]['Answer 4']
    row_ans4_embd = model.encode(row_ans4)
    compare_embedd.append(row_ans4_embd)

    row_ans5 = us_election_qa.iloc[i]['Answer 5']
    row_ans5_embd = model.encode(row_ans5)
    compare_embedd.append(row_ans5_embd)

    row_ans6 = us_election_qa.iloc[i]['Answer 6']
    row_ans6_embd = model.encode(row_ans6)
    compare_embedd.append(row_ans6_embd)

    row_ans7 = us_election_qa.iloc[i]['Answer 7']
    row_ans7_embd = model.encode(row_ans7)
    compare_embedd.append(row_ans7_embd)

    row_ans8 = us_election_qa.iloc[i]['Answer 8']
    row_ans8_embd = model.encode(row_ans8)
    compare_embedd.append(row_ans8_embd)

    row_ans9 = us_election_qa.iloc[i]['Answer 9']
    row_ans9_embd = model.encode(row_ans9)
    compare_embedd.append(row_ans9_embd)

    row_ans10 = us_election_qa.iloc[i]['Answer 10']
    row_ans10_embd = model.encode(row_ans10)
    compare_embedd.append(row_ans10_embd)

similarities = cosine_similarity([row_text_embd], compare_embedd)
print('pairwise dense output:\n {}\n'.format(similarities))