# Source: https://www.analyticsvidhya.com/blog/2018/04/a-comprehensive-guide-to-understand-and-implement-text-classification-in-python/
import json
import pandas as pd
import string
import os
import numpy as np
from textblob import TextBlob           #python -m pip install textblob
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from readability import Readability     #pip install py-readability-metrics
                                        #python -m nltk.downloader punkt
import spacy
import language_tool_python
from lda_data_processing import printProgressBar             #python -m pip install language_tool_python
from octis_topicmodeling import train_save_neurallda
from text_processing import text_preprocessing

nlp = spacy.load("en_core_web_sm")      #python3 -m spacy download en_core_web_sm

pos_family = {
    'noun': ['NN', 'NNS', 'NNP', 'NNPS'],
    'pron': ['PRP', 'PRP$', 'WP', 'WP$'],
    'verb': ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'],
    'adj': ['JJ', 'JJR', 'JJS'],
    'adv': ['RB', 'RBR', 'RBS', 'WRB']
}

# function to check and get the part of speech tag count of a words in a given sentence
def check_pos_tag(x, flag):
    cnt = 0
    try:
        wiki = textblob.TextBlob(x)
        for tup in wiki.tags:
            ppo = list(tup)[1]
            if ppo in pos_family[flag]:
                cnt += 1
    except:
        pass
    return cnt


def readbility_Score(trainDF, selected_col):
    result = pd.DataFrame()
    for i in range(trainDF.shape[0]):
        try:
            temp_result = []
            r = Readability(trainDF[selected_col].iloc[i])

            flesch_kincaid = r.flesch_kincaid()
            temp_result.append(flesch_kincaid.score)

            flesch = r.flesch()
            temp_result.append(flesch.score)

            gunning_fog = r.gunning_fog()
            temp_result.append(gunning_fog.score)

            coleman_liau = r.coleman_liau()
            temp_result.append(coleman_liau.score)

            dale_chall = r.dale_chall()
            temp_result.append(dale_chall.score)

            ari = r.ari()
            temp_result.append(ari.score)

            linsear_write = r.linsear_write()
            temp_result.append(linsear_write.score)

            spache = r.spache()
            temp_result.append(spache.score)

            temp_result = pd.DataFrame([temp_result],
                                       columns=['flesch kincaid score', 'flesch score', 'gunning fog score',
                                                'coleman liau score', \
                                                'dale chall score', 'ari score', 'linsear write score', 'spache score'])
        except:
            temp_result = pd.DataFrame([[0] * 8],
                                       columns=['flesch kincaid score', 'flesch score', 'gunning fog score',
                                                'coleman liau score', \
                                                'dale chall score', 'ari score', 'linsear write score', 'spache score'])

        result = pd.concat([result, temp_result], axis=0)
    result = result.reset_index(drop=True)
    return result


def count_ent(trainDF, selected_col):
    ner_count = []
    for i in range(trainDF.shape[0]):
        doc = nlp(trainDF[selected_col].iloc[i])
        ner = []
        if doc.ents:
            for ent in doc.ents:
                ner.append(ent.text)
        #else:
        #    print('NO EN')
        ner_count.append(len(ner))
    ner_count = pd.DataFrame(ner_count, columns=['NER Count'])
    return ner_count


tool = language_tool_python.LanguageTool('en-US')
def count_grammar_error(df):
    x = tool.check(df)
    return len(x)

def wiki_feature_extraction(inputPath='./data/chatgpt_generated_wiki_data_1_5000.csv', source=0):
    if source == 1:
        selected_col = 'GPT_Generated_Text'
        lda_dataset_path = './data/wiki_5k_chatgpt'
        lda_model_folder = './model/lda_wiki_5k_chatgpt'
    else:
        selected_col = 'Text'
        lda_dataset_path = './data/wiki_5k_gt'
        lda_model_folder = './model/lda_wiki_5k_gt'
    trainDF = pd.read_csv(inputPath)
    #trainDF = trainDF.iloc[0:10, :]
    indexes = trainDF.index.values  
    trainDF.insert( 0, column="indexes", value = indexes)
    
    # Removing rows having texts less than 100 words
    trainDF[selected_col].replace('', np.nan, inplace=True)
    trainDF.dropna(subset=[selected_col], inplace=True)
    trainDF["char_count"] = trainDF[selected_col].apply(lambda x: len(x))
    trainDF = trainDF[(trainDF["char_count"] >= 100) & (~trainDF[selected_col].str.startswith('http://'))]
    print('step 1: ', trainDF.shape)

    # Text,GPT_Generated_Text
    trainDF['word_count'] = trainDF[selected_col].str.split().apply(len)
    trainDF['word_density'] = trainDF['char_count'] / (trainDF['word_count'] + 1)
    trainDF['punctuation_count'] = trainDF[selected_col].apply(lambda x: len("".join(_ for _ in x if _ in string.punctuation)))
    trainDF['title_word_count'] = trainDF[selected_col].apply(lambda x: len([wrd for wrd in x.split() if wrd.istitle()]))
    trainDF['upper_case_word_count'] = trainDF[selected_col].apply(lambda x: len([wrd for wrd in x.split() if wrd.isupper()]))
    trainDF['noun_count'] = trainDF[selected_col].apply(lambda x: check_pos_tag(x, 'noun'))
    trainDF['verb_count'] = trainDF[selected_col].apply(lambda x: check_pos_tag(x, 'verb'))
    trainDF['adj_count'] = trainDF[selected_col].apply(lambda x: check_pos_tag(x, 'adj'))
    trainDF['adv_count'] = trainDF[selected_col].apply(lambda x: check_pos_tag(x, 'adv'))
    trainDF['pron_count'] = trainDF[selected_col].apply(lambda x: check_pos_tag(x, 'pron'))
    print('step 2 char_count: ', len(trainDF['char_count']), len(trainDF['word_count']))

    vocab = []
    with open('./data/wiki_5k_gt/vocabulary.txt', 'r') as vocabulary_file:
        for line in vocabulary_file:
            if line.strip() not in vocab:
                vocab.append(line.strip())
    with open('./data/wiki_5k_chatgpt/vocabulary.txt', 'r') as vocabulary_file:
        for line in vocabulary_file:
            if line.strip() not in vocab:
                vocab.append(line.strip())
    count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}', vocabulary=vocab)
    #text_data = pd.concat([trainDF['Text'], trainDF['GPT_Generated_Text']])
    #text_data.dropna(inplace=True)
    #count_vect.fit(text_data)
    #count_vect.fit(trainDF[['Text', 'GPT_Generated_Text']])
    vectorizer_count = count_vect.transform(trainDF[selected_col])
    vectorizer_count = pd.DataFrame(vectorizer_count.toarray())
    print('step 3 vectorizer_count.shape: ', vectorizer_count.shape)

    # word level tf-idf
    tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
    tfidf_vect.fit(trainDF[selected_col])
    xtrain_tfidf = tfidf_vect.transform(trainDF[selected_col])
    tfidf_word = pd.DataFrame(xtrain_tfidf.toarray())
    print('step 4 tfidf_word.shape: ', tfidf_word.shape)

    # ngram level tf-idf
    tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2, 3), max_features=5000)
    tfidf_vect_ngram.fit(trainDF[selected_col])
    trainDF_tfidf_ngram = tfidf_vect_ngram.transform(trainDF[selected_col])
    tfidf_ngram = pd.DataFrame(trainDF_tfidf_ngram.toarray())
    print('step 5 tfidf_ngram.shape: ', tfidf_ngram.shape)
    #
    # # characters level tf-idf
    tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2, 3),
                                            max_features=5000)
    tfidf_vect_ngram_chars.fit(trainDF[selected_col])
    trainDF_tfidf_ngram_chars = tfidf_vect_ngram_chars.transform(trainDF)
    tfidf_ngram_chars = pd.DataFrame(trainDF_tfidf_ngram_chars.toarray())
    print('step 6 tfidf_ngram_chars.shape: ', tfidf_ngram_chars.shape)

    # Get readability score
    readbility = readbility_Score(trainDF, selected_col)
    print('step 7 readbility.shape: ', readbility.shape)

    # Get NER count
    nercount = count_ent(trainDF, selected_col)
    print('step 8 nercount.shape: ', nercount.shape)

    # Grammar check
    trainDF['text_error_length'] = trainDF[selected_col].apply(count_grammar_error)

    #extract topic modeling features (20 columns)
    _, _, lda_df = train_save_neurallda(dataset_path=lda_dataset_path, out_folder=lda_model_folder)

    #save the corresponding texts for the wiki dataset
    out_folder = './data/wiki_features/'
    os.makedirs(out_folder, exist_ok=True)
    wiki_text_df = trainDF[["indexes", selected_col]]
    wiki_text_df.rename(columns={selected_col : 'Text'})
    wiki_text_df.to_csv(out_folder + '/wiki_text_{0}.csv'.format('GT' if source==0 else 'ChatGPT'))

    #save the handcrafted features of the wiki dataset
    feat_df = trainDF[["indexes", 'word_count', 'char_count', 'word_density', 'punctuation_count', 'title_word_count', 
                       'upper_case_word_count', 'noun_count', 'verb_count', 'adj_count', 'adv_count', 'pron_count', 'text_error_length']]
    print('step 9 nercount.shape: ', feat_df.shape)
    #tfidf_ngram_chars has the shape (15, 5000) incompatible with the dataset => skipped
    feature_set = pd.concat([feat_df.reset_index(drop=True), readbility.reset_index(drop=True), nercount.reset_index(drop=True), 
                             lda_df.reset_index(drop=True), vectorizer_count.reset_index(drop=True), tfidf_word.reset_index(drop=True), 
                             tfidf_ngram.reset_index(drop=True)], axis=1)
    print(feature_set.shape)
    feature_set.to_csv(out_folder + '/wiki_features_{0}.csv'.format('GT' if source==0 else 'ChatGPT'))

def us_election_lda_data_processing(inputPath='./data/chatgpt_generated_us_election_2024_questions_answers_combine.csv'):
    trainDF = pd.read_csv(inputPath)
    for i in range(11):
        selected_col = 'Text' if i == 0 else 'Question ' + str(i)
        out_folder = './data/uselection_' + ('gt' if i == 0 else 'q' + str(i))
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
                document_label = str(index) + '.' + selected_col
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
        metadata['info'] = inputPath
        metadata['labels'] = ''
        metadata['total_labels'] = len(labelList)
        with open(out_folder + '/metadata.json', 'w') as fp:
            json.dump(metadata, fp)

def us_election_feature_extraction(inputPath='./data/chatgpt_generated_us_election_2024_questions_answers_combine.csv'):
    trainDF = pd.read_csv(inputPath)
    indexes = trainDF.index.values  
    trainDF.insert( 0, column="indexes", value = indexes)
    vocab = []
    for i in range(11):
        lda_dataset_path = './data/uselection_' + ('gt' if i == 0 else 'q' + str(i))
        with open(lda_dataset_path + '/vocabulary.txt', 'r') as vocabulary_file:
            for line in vocabulary_file:
                if line.strip() not in vocab:
                    vocab.append(line.strip())
    for i in range(11):
        selected_col = 'Text' if i == 0 else 'Question ' + str(i)
        lda_dataset_path = './data/uselection_' + ('gt' if i == 0 else 'q' + str(i))
        lda_model_folder = './model/uselection_' + ('gt' if i == 0 else 'q' + str(i))
        # Removing rows having texts less than 100 words
        trainDF[selected_col].replace('', np.nan, inplace=True)
        trainDF.dropna(subset=[selected_col], inplace=True)
        trainDF["char_count"] = trainDF[selected_col].apply(lambda x: len(x))
        trainDF = trainDF[(trainDF["char_count"] >= 100) & (~trainDF[selected_col].str.startswith('http://'))]
        print('step 1: ', trainDF.shape)

        # Text,GPT_Generated_Text
        trainDF['word_count'] = trainDF[selected_col].str.split().apply(len)
        trainDF['word_density'] = trainDF['char_count'] / (trainDF['word_count'] + 1)
        trainDF['punctuation_count'] = trainDF[selected_col].apply(lambda x: len("".join(_ for _ in x if _ in string.punctuation)))
        trainDF['title_word_count'] = trainDF[selected_col].apply(lambda x: len([wrd for wrd in x.split() if wrd.istitle()]))
        trainDF['upper_case_word_count'] = trainDF[selected_col].apply(lambda x: len([wrd for wrd in x.split() if wrd.isupper()]))
        trainDF['noun_count'] = trainDF[selected_col].apply(lambda x: check_pos_tag(x, 'noun'))
        trainDF['verb_count'] = trainDF[selected_col].apply(lambda x: check_pos_tag(x, 'verb'))
        trainDF['adj_count'] = trainDF[selected_col].apply(lambda x: check_pos_tag(x, 'adj'))
        trainDF['adv_count'] = trainDF[selected_col].apply(lambda x: check_pos_tag(x, 'adv'))
        trainDF['pron_count'] = trainDF[selected_col].apply(lambda x: check_pos_tag(x, 'pron'))
        print('step 2 char_count: ', len(trainDF['char_count']), len(trainDF['word_count']))

        count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}', vocabulary=vocab)
        vectorizer_count = count_vect.transform(trainDF[selected_col])
        vectorizer_count = pd.DataFrame(vectorizer_count.toarray())
        print('step 3 vectorizer_count.shape: ', vectorizer_count.shape)

        # word level tf-idf
        tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000, vocabulary=vocab)
        tfidf_vect.fit(trainDF[selected_col])
        xtrain_tfidf = tfidf_vect.transform(trainDF[selected_col])
        tfidf_word = pd.DataFrame(xtrain_tfidf.toarray())
        print('step 4 tfidf_word.shape: ', tfidf_word.shape)

        # ngram level tf-idf
        tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2, 3), max_features=5000, vocabulary=vocab)
        tfidf_vect_ngram.fit(trainDF[selected_col])
        trainDF_tfidf_ngram = tfidf_vect_ngram.transform(trainDF[selected_col])
        tfidf_ngram = pd.DataFrame(trainDF_tfidf_ngram.toarray())
        print('step 5 tfidf_ngram.shape: ', tfidf_ngram.shape)
        #
        # # characters level tf-idf
        tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2, 3),
                                                max_features=5000, vocabulary=vocab)
        tfidf_vect_ngram_chars.fit(trainDF[selected_col])
        trainDF_tfidf_ngram_chars = tfidf_vect_ngram_chars.transform(trainDF)
        tfidf_ngram_chars = pd.DataFrame(trainDF_tfidf_ngram_chars.toarray())
        print('step 6 tfidf_ngram_chars.shape: ', tfidf_ngram_chars.shape)

        # Get readability score
        readbility = readbility_Score(trainDF, selected_col)
        print('step 7 readbility.shape: ', readbility.shape)

        # Get NER count
        nercount = count_ent(trainDF, selected_col)
        print('step 8 nercount.shape: ', nercount.shape)

        # Grammar check
        trainDF['text_error_length'] = trainDF[selected_col].apply(count_grammar_error)

        #extract topic modeling features (20 columns)
        _, _, lda_df = train_save_neurallda(dataset_path=lda_dataset_path, out_folder=lda_model_folder)

        #save the corresponding texts for the uselection dataset
        out_folder = './data/uselection_features/'
        os.makedirs(out_folder, exist_ok=True)
        uselection_text_df = trainDF[["indexes", selected_col]]
        uselection_text_df.rename(columns={selected_col : 'Text'})
        uselection_text_df.to_csv(out_folder + '/uselection_text_{0}.csv'.format('GT' if i==0 else 'ChatGPT'+str(i)))

        #extract word features      
        feat_df = trainDF[['indexes', 'word_count', 'char_count', 'word_density', 'punctuation_count', 'title_word_count', 
                        'upper_case_word_count', 'noun_count', 'verb_count', 'adj_count', 'adv_count', 'pron_count', 'text_error_length']]
        print('step 9 nercount.shape: ', feat_df.shape)
        #tfidf_ngram_chars has the shape (15, 5000) incompatible with the dataset => skipped
        feature_set = pd.concat([feat_df.reset_index(drop=True), vectorizer_count.reset_index(drop=True), tfidf_word.reset_index(drop=True), 
                                tfidf_ngram.reset_index(drop=True), readbility.reset_index(drop=True), nercount.reset_index(drop=True), 
                                lda_df.reset_index(drop=True)], axis=1)
        print(feature_set.shape)
        feature_set.to_csv(out_folder + '/uselection_features_{0}.csv'.format('GT' if i==0 else 'ChatGPT'+str(i)))

if __name__ == "__main__":
    #wiki_feature_extraction(source=0)   #process TEXT column in './data/chatgpt_generated_wiki_data_1_5000.csv'
    #wiki_feature_extraction(source=1)   #process GPT_Generated_Text column in './data/chatgpt_generated_wiki_data_1_5000.csv'
    #us_election_lda_data_processing()   #preprocess data for the us election 2024 dataset => to perform topic modeling
    us_election_feature_extraction()     #extract features for the us election dataset