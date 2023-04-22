#example run command: python .\code\text_features\lda_data_processing.py --input_file=.\data\chatgpt_generated_wiki_data_1_5000.csv --source=0 --out_folder=.\data\wiki_5k_gt

# Importing Libraries
import pandas as pd
import numpy as np
import string, random
from text_processing import text_preprocessing
import argparse

parser = argparse.ArgumentParser("lda data processing")
parser.add_argument("--input_file", type=str, help="input file path", default='../../data/chatgpt_generated_wiki_data_1_5000.csv')
parser.add_argument("--source", type=int, help="selected column: 0-Text, 1-GPT_Generated_Text", default=0)
parser.add_argument("--out_folder", type=str, help="output folder", default='../data')
args = parser.parse_args()

input_file = args.input_file
source = args.source
out_folder = args.out_folder

trainDF = pd.read_csv(input_file)
#trainDF = trainDF.iloc[0:10, :]

# Removing rows having texts less than 100 words
trainDF['Text'].replace('', np.nan, inplace=True)
trainDF.dropna(subset=['Text'], inplace=True)

trainDF["word_count"] = trainDF["Text"].apply(lambda x: len(x))
trainDF = trainDF[(trainDF["word_count"] >= 100)]

#loop for each Text on each row to collect data:
vocab = []
documentList = []
partitionList = []
cleanedTextList = []
labelList = []
sum_word_len = 0
for index, row in trainDF.iterrows():
    if index+1 % 100 == 0:
        print( 'process ' + (index+1)/trainDF.shape[0]*100 + '% (' + (index+1) + '/' + trainDF.shape[0] + ')' )
    # determine the document label for this data row
    document_label = row['Page title'] + ' ' + row['Section title']
    documentList.append(document_label)
    # determine the train/test label with the ratio 80:20
    rand = random.random()
    train_test_label = "train" if rand <= 0.8 else "test"
    partitionList.append(train_test_label)
    # preprocess the TEXT attribute
    if source == 1:
        Processed_Content = text_preprocessing(row['GPT_Generated_Text']) #Cleaned text of Content attribute after pre-processing
    else:    
        Processed_Content = text_preprocessing(row['Text']) #Cleaned text of Content attribute after pre-processing
    cleanedTextList.append(Processed_Content)
    sum_word_len += len(Processed_Content)
    for word in Processed_Content:
        if word not in vocab:
            vocab.append(word)
    # determine the label
    labelList.append(row['Page title'])
trainDF['documentList'] = documentList
trainDF['partitionList'] = partitionList
trainDF['labelList'] = labelList
trainDF['cleanedTextList'] = cleanedTextList

# make output directory if not exist
import os
os.makedirs(out_folder, exist_ok=True)

# write vocabulary text file
with open(out_folder + '/vocabulary.txt', 'w') as f:
    for line in vocab:
        f.write(f"{line}\n")

# export document, partition, label columns to corpus.tsv file
import csv
corpusDF = trainDF[['documentList', 'partitionList', 'labelList']]
corpusDF.to_csv(out_folder + '/corpus.tsv', sep='\t', header=None)

#export metadata.json
metadata = {}
metadata['total_documents'] = len(documentList)
metadata['words_document_mean'] = sum_word_len/len(documentList)
metadata['vocabulary_length'] = len(vocab)
metadata['last-training-doc'] = 0
metadata['last-validation-doc'] = 0
metadata['preprocessing-info'] = "Steps:\n  remove_punctuation\n  lemmatization\n  remove_stopwords\n  filter_words\n  remove_docs\nParameters:\n  \
    removed documents with less than 100 words"
metadata['info'] = input_file
metadata['labels'] = ''
metadata['total_labels'] = len(labelList)
import json
with open(out_folder + '/metadata.json', 'w') as fp:
    json.dump(metadata, fp)