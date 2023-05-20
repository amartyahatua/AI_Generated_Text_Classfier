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
        try:
            ai_gen_text = []
            row_text = us_election_qa.iloc[i]['Text']
            row_text_embd = model.encode(row_text)

            ai_gen_text.append(us_election_qa.iloc[i]['Answer 1']) if us_election_qa.iloc[i][
                                                                          'Answer 1'] is not None else \
            us_election_qa.iloc[i]['Title']
            ai_gen_text.append(us_election_qa.iloc[i]['Answer 2']) if us_election_qa.iloc[i][
                                                                          'Answer 2'] is not None else \
            us_election_qa.iloc[i]['Title']
            ai_gen_text.append(us_election_qa.iloc[i]['Answer 3']) if us_election_qa.iloc[i][
                                                                          'Answer 3'] is not None else \
            us_election_qa.iloc[i]['Title']
            ai_gen_text.append(us_election_qa.iloc[i]['Answer 4']) if us_election_qa.iloc[i][
                                                                          'Answer 4'] is not None else \
            us_election_qa.iloc[i]['Title']
            ai_gen_text.append(us_election_qa.iloc[i]['Answer 5']) if us_election_qa.iloc[i][
                                                                          'Answer 5'] is not None else \
            us_election_qa.iloc[i]['Title']
            ai_gen_text.append(us_election_qa.iloc[i]['Answer 6']) if us_election_qa.iloc[i][
                                                                          'Answer 6'] is not None else \
            us_election_qa.iloc[i]['Title']
            ai_gen_text.append(us_election_qa.iloc[i]['Answer 7']) if us_election_qa.iloc[i][
                                                                          'Answer 7'] is not None else \
            us_election_qa.iloc[i]['Title']
            ai_gen_text.append(us_election_qa.iloc[i]['Answer 8']) if us_election_qa.iloc[i][
                                                                          'Answer 8'] is not None else \
            us_election_qa.iloc[i]['Title']
            ai_gen_text.append(us_election_qa.iloc[i]['Answer 9']) if us_election_qa.iloc[i][
                                                                          'Answer 9'] is not None else \
            us_election_qa.iloc[i]['Title']
            ai_gen_text.append(us_election_qa.iloc[i]['Answer 10']) if us_election_qa.iloc[i][
                                                                           'Answer 10'] is not None else \
            us_election_qa.iloc[i]['Title']
            compare_embedd = model.encode(ai_gen_text)

            similarities_temp = pd.DataFrame(cosine_similarity([row_text_embd], compare_embedd),
                                             columns=['Cos Similarity 1', 'Cos Similarity 2', 'Cos Similarity 3',
                                                      'Cos Similarity 4',
                                                      'Cos Similarity 5', 'Cos Similarity 6', 'Cos Similarity 7',
                                                      'Cos Similarity 8',
                                                      'Cos Similarity 9', 'Cos Similarity 10'])
            similarities = pd.concat([similarities_temp, similarities], axis=0)
        except:
            similarities_temp = pd.DataFrame([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                                             columns=['Cos Similarity 1', 'Cos Similarity 2', 'Cos Similarity 3',
                                                      'Cos Similarity 4',
                                                      'Cos Similarity 5', 'Cos Similarity 6', 'Cos Similarity 7',
                                                      'Cos Similarity 8',
                                                      'Cos Similarity 9', 'Cos Similarity 10'])
            similarities = pd.concat([similarities_temp, similarities], axis=0)
            print('Here')
            continue

    similarities = similarities.reset_index(drop=True)
    return  similarities


def jaccard_similarity(doc1, doc2):
    # List the unique words in a document
    words_doc1 = set(doc1.lower().split())
    words_doc2 = set(doc2.lower().split())

    # Find the intersection of words list of doc1 & doc2
    intersection = words_doc1.intersection(words_doc2)

    # Find the union of words list of doc1 & doc2
    union = words_doc1.union(words_doc2)

    # Calculate Jaccard similarity score
    # using length of intersection set divided by length of union set
    return float(len(intersection)) / len(union)


def find_jaccard_similarity():
    similarities = pd.DataFrame()
    us_election_qa = pd.read_csv('../../data/chatgpt_generated_us_election_2024_questions_answers_combine.csv')
    for i in tqdm(range(us_election_qa.shape[0])):
        try:
            temp_similarity = []
            ai_gen_text = []

            row_text = us_election_qa.iloc[i]['Text']
            us_election_qa = us_election_qa.replace(np.nan, '', regex=True)

            ai_gen_text.append(us_election_qa.iloc[i]['Answer 1']) if us_election_qa.iloc[i][
                                                                          'Answer 1'] is not None else \
                us_election_qa.iloc[i]['Title']
            ai_gen_text.append(us_election_qa.iloc[i]['Answer 2']) if us_election_qa.iloc[i][
                                                                          'Answer 2'] is not None else \
                us_election_qa.iloc[i]['Title']
            ai_gen_text.append(us_election_qa.iloc[i]['Answer 3']) if us_election_qa.iloc[i][
                                                                          'Answer 3'] is not None else \
                us_election_qa.iloc[i]['Title']
            ai_gen_text.append(us_election_qa.iloc[i]['Answer 4']) if us_election_qa.iloc[i][
                                                                          'Answer 4'] is not None else \
                us_election_qa.iloc[i]['Title']
            ai_gen_text.append(us_election_qa.iloc[i]['Answer 5']) if us_election_qa.iloc[i][
                                                                          'Answer 5'] is not None else \
                us_election_qa.iloc[i]['Title']
            ai_gen_text.append(us_election_qa.iloc[i]['Answer 6']) if us_election_qa.iloc[i][
                                                                          'Answer 6'] is not None else \
                us_election_qa.iloc[i]['Title']
            ai_gen_text.append(us_election_qa.iloc[i]['Answer 7']) if us_election_qa.iloc[i][
                                                                          'Answer 7'] is not None else \
                us_election_qa.iloc[i]['Title']
            ai_gen_text.append(us_election_qa.iloc[i]['Answer 8']) if us_election_qa.iloc[i][
                                                                          'Answer 8'] is not None else \
                us_election_qa.iloc[i]['Title']
            ai_gen_text.append(us_election_qa.iloc[i]['Answer 9']) if us_election_qa.iloc[i][
                                                                          'Answer 9'] is not None else \
                us_election_qa.iloc[i]['Title']
            ai_gen_text.append(us_election_qa.iloc[i]['Answer 10']) if us_election_qa.iloc[i][
                                                                           'Answer 10'] is not None else \
                us_election_qa.iloc[i]['Title']

            for i in range(len(ai_gen_text)):
                try:
                    temp_similarity.append(jaccard_similarity(row_text, ai_gen_text[i]))
                except:
                    temp_similarity.append(0)
                    print('Here')

            temp_similarity = pd.DataFrame([temp_similarity],
                                           columns=['JAC Similarity 1', 'JAC Similarity 2', 'JAC Similarity 3',
                                                    'JAC Similarity 4', 'JAC Similarity 5', 'JAC Similarity 6',
                                                    'JAC Similarity 7',
                                                    'JAC Similarity 8', 'JAC Similarity 9', 'JAC Similarity 10'])
            similarities = pd.concat([temp_similarity, similarities], axis=0)
        except:
            continue

    return similarities


cosine_similarity = cosine_similarity_sentence_transformer()
cosine_similarity = cosine_similarity.reset_index(drop=True)

jaccard_similarity = find_jaccard_similarity()
jaccard_similarity = jaccard_similarity.reset_index(drop=True)

us_election_qa_smilarity = pd.concat(
    [us_election_qa[['Title', 'Text', 'Summary', 'Keywords', 'Question 1', 'Answer 1']],
     cosine_similarity['Cos Similarity 1'], jaccard_similarity['JAC Similarity 1'],
     us_election_qa[['Question 2', 'Answer 2']],
     cosine_similarity['Cos Similarity 2'], jaccard_similarity['JAC Similarity 2'],
     us_election_qa[['Question 3', 'Answer 3']],
     cosine_similarity['Cos Similarity 3'], jaccard_similarity['JAC Similarity 3'],
     us_election_qa[['Question 4', 'Answer 4']],
     cosine_similarity['Cos Similarity 4'], jaccard_similarity['JAC Similarity 4'],
     us_election_qa[['Question 5', 'Answer 5']],
     cosine_similarity['Cos Similarity 5'], jaccard_similarity['JAC Similarity 5'],
     us_election_qa[['Question 6', 'Answer 6']],
     cosine_similarity['Cos Similarity 6'], jaccard_similarity['JAC Similarity 6'],
     us_election_qa[['Question 7', 'Answer 7']],
     cosine_similarity['Cos Similarity 7'], jaccard_similarity['JAC Similarity 7'],
     us_election_qa[['Question 8', 'Answer 8']],
     cosine_similarity['Cos Similarity 8'], jaccard_similarity['JAC Similarity 8'],
     us_election_qa[['Question 9', 'Answer 9']],
     cosine_similarity['Cos Similarity 9'], jaccard_similarity['JAC Similarity 9'],
     us_election_qa[['Question 10', 'Answer 10']],
     cosine_similarity['Cos Similarity 10'], jaccard_similarity['JAC Similarity 10'],], axis=1)

# print(us_election_qa_smilarity.shape)
# cosine_similarity.to_csv('../../data/chatgpt_generated_us_election_2024_questions_answers_cosine_similarity.csv',
#                                 index=False)
#
# jaccard_similarity.to_csv('../../data/chatgpt_generated_us_election_2024_questions_answers_jaccard_similarity.csv',
#                                 index=False)

us_election_qa_smilarity.to_csv('../../data/chatgpt_generated_us_election_2024_questions_answers_similarity.csv',
                              index=False)
