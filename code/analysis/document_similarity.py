# source:https://medium.com/@adriensieg/text-similarities-da019229c894

import pandas as pd
import numpy as np
from nltk.corpus import stopwords
# import nltk

# nltk.download('stopwords')
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

model = SentenceTransformer('bert-base-nli-mean-tokens')

us_election_qa = pd.read_csv('../../data/chatgpt_generated_us_election_2024_questions_answers_combine.csv')


def cosine_similarity_sentence_transformer():
    similarities = pd.DataFrame()
    for i in tqdm(range(us_election_qa.shape[0])):
        ai_gen_text = []
        row_text = us_election_qa.iloc[i]['Text']
        row_text_embd = model.encode(row_text)

        ai_gen_text.append(us_election_qa.iloc[i]['Answer 1']) if us_election_qa.iloc[i][
                                                                      'Answer 1'] is not None else None
        ai_gen_text.append(us_election_qa.iloc[i]['Answer 2']) if us_election_qa.iloc[i][
                                                                      'Answer 2'] is not None else None
        ai_gen_text.append(us_election_qa.iloc[i]['Answer 3']) if us_election_qa.iloc[i][
                                                                      'Answer 3'] is not None else None
        ai_gen_text.append(us_election_qa.iloc[i]['Answer 4']) if us_election_qa.iloc[i][
                                                                      'Answer 4'] is not None else None
        ai_gen_text.append(us_election_qa.iloc[i]['Answer 5']) if us_election_qa.iloc[i][
                                                                      'Answer 5'] is not None else None
        ai_gen_text.append(us_election_qa.iloc[i]['Answer 6']) if us_election_qa.iloc[i][
                                                                      'Answer 6'] is not None else None
        ai_gen_text.append(us_election_qa.iloc[i]['Answer 7']) if us_election_qa.iloc[i][
                                                                      'Answer 7'] is not None else None
        ai_gen_text.append(us_election_qa.iloc[i]['Answer 8']) if us_election_qa.iloc[i][
                                                                      'Answer 8'] is not None else None
        ai_gen_text.append(us_election_qa.iloc[i]['Answer 9']) if us_election_qa.iloc[i][
                                                                      'Answer 9'] is not None else None
        ai_gen_text.append(us_election_qa.iloc[i]['Answer 10']) if us_election_qa.iloc[i][
                                                                       'Answer 10'] is not None else None
        compare_embedd = model.encode(ai_gen_text)

        similarities_temp = pd.DataFrame(cosine_similarity([row_text_embd], compare_embedd),
                                         columns=['Cos Similarity 1', 'Cos Similarity 2', 'Cos Similarity 3',
                                                  'Cos Similarity 4',
                                                  'Cos Similarity 5', 'Cos Similarity 6', 'Cos Similarity 7',
                                                  'Cos Similarity 8',
                                                  'Cos Similarity 9', 'Cos Similarity 10'])
        similarities = pd.concat([similarities_temp, similarities], axis=0)

    similarities = similarities.reset_index(drop=True)

    us_election_qa_smilarity = pd.concat(
        [us_election_qa[['Title', 'Text', 'Summary', 'Keywords', 'Question 1', 'Answer 1']],
         similarities['Cos Similarity 1'],
         us_election_qa[['Question 2', 'Answer 2']],
         similarities['Cos Similarity 2'],
         us_election_qa[['Question 3', 'Answer 3']],
         similarities['Cos Similarity 3'],
         us_election_qa[['Question 4', 'Answer 4']],
         similarities['Cos Similarity 4'],
         us_election_qa[['Question 5', 'Answer 5']],
         similarities['Cos Similarity 5'],
         us_election_qa[['Question 6', 'Answer 6']],
         similarities['Cos Similarity 6'],
         us_election_qa[['Question 7', 'Answer 7']],
         similarities['Cos Similarity 7'],
         us_election_qa[['Question 8', 'Answer 8']],
         similarities['Cos Similarity 8'],
         us_election_qa[['Question 9', 'Answer 9']],
         similarities['Cos Similarity 9'],
         us_election_qa[['Question 10', 'Answer 10']],
         similarities['Cos Similarity 10']], axis=1)

    print(us_election_qa_smilarity.shape)
    us_election_qa_smilarity.to_csv('../../data/chatgpt_generated_us_election_2024_questions_answers_similarity.csv', index=False)


cosine_similarity_sentence_transformer()
