# source:https://medium.com/@adriensieg/text-similarities-da019229c894

import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


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

df =  pd.read_csv('../../data/wiki_features_similarity_no_tfidf.csv')
first_bin = 0
second_bin = 0
third_bin = 0
fourth_bin = 0
fifth_bin = 0

for i in df['Cosine similarity']:
    if i >= -1 and i <= -0.6:
        first_bin += 1
    elif i > -0.6 and i <= 0.2:
        second_bin += 1
    elif i > 0.2 and i <= 0.6:
        third_bin += 1
    elif i > 0.6 and i <= 1:
        fourth_bin += 1
print("Without TFIDF")

print('First bin: ', first_bin)
print('Second bin: ', second_bin)
print('Third bin: ', third_bin)
print('Fourth bin: ',fourth_bin)

print("{:.2f} % document with Cosine similarity -1 to -0.6".format(first_bin/df['Cosine similarity'].shape[0]*100))
print("{:.2f} % document has Cosine similarity -0.6 to -0.2".format(second_bin/df['Cosine similarity'].shape[0]*100))
print("{:.2f} % document has Cosine similarity -0.2 to 0.6".format(third_bin/df['Cosine similarity'].shape[0]*100))
print("{:.2f} % document has Cosine similarity 0.6 to 1".format(fourth_bin/df['Cosine similarity'].shape[0]*100))

more_than_50 = 0
less_than_50 = 0

for i in df['Cosine similarity']:
    if i >= -1 and i <= 0:
        less_than_50 += 1
    elif i > 0 and i <= 1:
        more_than_50 += 1

print('First bin: ', less_than_50/df['Cosine similarity'].shape[0])
print('Second bin: ', more_than_50/df['Cosine similarity'].shape[0])


print("With TFIDF")
df =  pd.read_csv('../../data/wiki_features_similarity_with_tfidf.csv')
first_bin = 0
second_bin = 0
third_bin = 0
fourth_bin = 0
fifth_bin = 0

for i in df['Cosine similarity']:
    if i >= -1 and i <= -0.6:
        first_bin += 1
    elif i > -0.6 and i <= 0.2:
        second_bin += 1
    elif i > 0.2 and i <= 0.6:
        third_bin += 1
    elif i > 0.6 and i <= 1:
        fourth_bin += 1

print('First bin: ', first_bin)
print('Second bin: ', second_bin)
print('Third bin: ', third_bin)
print('Fourth bin: ',fourth_bin)

print("{:.2f} % document with Cosine similarity -1 to -0.6".format(first_bin/df['Cosine similarity'].shape[0]*100))
print("{:.2f} % document has Cosine similarity -0.6 to -0.2".format(second_bin/df['Cosine similarity'].shape[0]*100))
print("{:.2f} % document has Cosine similarity -0.2 to 0.6".format(third_bin/df['Cosine similarity'].shape[0]*100))
print("{:.2f} % document has Cosine similarity 0.6 to 1".format(fourth_bin/df['Cosine similarity'].shape[0]*100))

more_than_50 = 0
less_than_50 = 0

for i in df['Cosine similarity']:
    if i >= -1 and i <= 0:
        less_than_50 += 1
    elif i > 0 and i <= 1:
        more_than_50 += 1

print('First bin: ', less_than_50/df['Cosine similarity'].shape[0])
print('Second bin: ', more_than_50/df['Cosine similarity'].shape[0])

