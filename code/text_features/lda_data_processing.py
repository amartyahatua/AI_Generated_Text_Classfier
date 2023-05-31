#example run command: python .\code\text_features\lda_data_processing.py --input_file=.\data\chatgpt_generated_wiki_data_1_5000.csv --source=0 --out_folder=.\data\wiki_5k_gt

# Importing Libraries
import pandas as pd
import numpy as np
import string, random
import json
import argparse
from text_processing import text_preprocessing

# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser("lda data processing")
    parser.add_argument("--input_file", type=str, help="input file path", default='./data/chatgpt_generated_wiki_data_1_5000.csv')
    parser.add_argument("--source", type=int, help="selected column: 0-Text, 1-GPT_Generated_Text", default=0)
    parser.add_argument("--out_folder", type=str, help="output folder", default='./data/wiki_5k_gt')
    args = parser.parse_args()

    input_file = args.input_file
    source = args.source
    out_folder = args.out_folder
    if source == 1:
        selected_col = 'GPT_Generated_Text'
    else:
        selected_col = 'Text'

    trainDF = pd.read_csv(input_file)
    #trainDF = trainDF.iloc[0:10, :]

    # Removing rows having texts less than 100 words
    trainDF[selected_col].replace('', np.nan, inplace=True)
    trainDF.dropna(subset=[selected_col], inplace=True)

    trainDF["word_count"] = trainDF[selected_col].apply(lambda x: len(x))
    trainDF = trainDF[(trainDF["word_count"] >= 100)]

    #loop for each Text on each row to collect data:
    vocab = []
    documentList = []
    partitionList = []
    labelList = []
    sum_word_len = 0
    count = 0
    totalRows = trainDF.shape[0]
    printProgressBar(count, totalRows, prefix = 'Progress:', suffix = 'Complete', length = 50)
    for index, row in trainDF.iterrows():
        count += 1 
        printProgressBar(count, totalRows, prefix = 'Progress:', suffix = 'Complete', length = 50)
        #if count % 100 == 0:
        #    print( 'process {0:0.2f}% ({1}/{2})'.format( (index+1)/trainDF.shape[0]*100, (index+1), trainDF.shape[0]) )
        # determine the document label for this data row 
        # preprocess the TEXT or ChatGPT response attribute (depending on the source argument)
        Processed_Content = text_preprocessing(row[selected_col])
        if len( str(Processed_Content).strip() ) > 0:
            documentList.append(' '.join(Processed_Content))
            sum_word_len += len(Processed_Content)
            for word in Processed_Content:
                if word not in vocab:
                    vocab.append(word)
            # determine the train/test label with the ratio 80:20
            #rand = random.random()
            train_test_label = "train" if count <= 0.8*totalRows else "val" if count <= 0.9*totalRows else "test"
            partitionList.append(train_test_label)
            # determine the label
            document_label = row['Page title'] + '.' + row['Section title']
            labelList.append(document_label)
    trainDF['documentList'] = documentList
    trainDF['partitionList'] = partitionList
    trainDF['labelList'] = labelList

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
    corpusDF.to_csv(out_folder + '/corpus.tsv', sep='\t', header=None, index=None)

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
    with open(out_folder + '/metadata.json', 'w') as fp:
        json.dump(metadata, fp)