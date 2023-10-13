# text processing utilitties credits: @techykajal
# #require libraries: nltk, autocorrect, bs4, unidecode
#pip install autocorrect nltk bs4 unidecode
# Importing Libraries
import pandas as pd
import re
import time
import nltk
import unidecode
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from autocorrect import Speller
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk import word_tokenize
import string
import timeit
stoplist = stopwords.words('english') 
stoplist = set(stoplist)
spell = Speller(lang='en')
w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()

def remove_newlines_tabs(text):
    """
    This function will remove all the occurrences of newlines, tabs, and combinations like: \\n, \\.    
    arguments:
        input_text: "text" of type "String".                     
    return:
        value: "text" after removal of newlines, tabs, \\n, \\ characters.        
    Example:
    Input : This is her \\ first day at this place.\n Please,\t Be nice to her.\\n
    Output : This is her first day at this place. Please, Be nice to her.     
    """    
    # Replacing all the occurrences of \n,\\n,\t,\\ with a space.
    Formatted_text = text.replace('\\n', ' ').replace('\n', ' ').replace('\t',' ').replace('\\', ' ').replace('. com', '.com')
    return Formatted_text

def strip_html_tags(text):
    """ 
    This function will remove all the occurrences of html tags from the text.    
    arguments:
        input_text: "text" of type "String".                     
    return:
        value: "text" after removal of html tags.        
    Example:
    Input : This is a nice place to live. <IMG>
    Output : This is a nice place to live.  
    """
    # Initiating BeautifulSoup object soup.
    soup = BeautifulSoup(text, "html.parser")
    # Get all the text other than html tags.
    stripped_text = soup.get_text(separator=" ")
    return stripped_text

def remove_links(text):
    """
    This function will remove all the occurrences of links.    
    arguments:
        input_text: "text" of type "String".                     
    return:
        value: "text" after removal of all types of links.        
    Example:
    Input : To know more about cats and food & website: catster.com  visit: https://catster.com//how-to-feed-cats
    Output : To know more about cats and food & website: visit:       
    """    
    # Removing all the occurrences of links that starts with https
    remove_https = re.sub(r'http\S+', '', text)
    # Remove all the occurrences of text that ends with .com
    remove_com = re.sub(r"\ [A-Za-z]*\.com", " ", remove_https)
    return remove_com

def remove_whitespace(text):
    """ This function will remove 
        extra whitespaces from the text
    arguments:
        input_text: "text" of type "String".                     
    return:
        value: "text" after extra whitespaces removed .        
    Example:
    Input : How   are   you   doing   ?
    Output : How are you doing ?        
    """
    pattern = re.compile(r'\s+') 
    Without_whitespace = re.sub(pattern, ' ', text)
    # There are some instances where there is no space after '?' & ')', 
    # So I am replacing these with one space so that It will not consider two words as one token.
    text = Without_whitespace.replace('?', ' ? ').replace(')', ') ')
    return text   

# Code for accented characters removal
def accented_characters_removal(text):
    # this is a docstring
    """
    The function will remove accented characters from the 
    text contained within the Dataset.       
    arguments:
        input_text: "text" of type "String".                     
    return:
        value: "text" with removed accented characters.        
    Example:
    Input : Málaga, àéêöhello
    Output : Malaga, aeeohello            
    """
    # Remove accented characters from text using unidecode.
    # Unidecode() - It takes unicode data & tries to represent it to ASCII characters. 
    text = unidecode.unidecode(text)
    return text

# Code for text lowercasing
def lower_casing_text(text):    
    """
    The function will convert text into lower case.    
    arguments:
         input_text: "text" of type "String".         
    return:
         value: text in lowercase         
    Example:
    Input : The World is Full of Surprises!
    Output : the world is full of surprises!    
    """
    # Convert text to lower case
    # lower() - It converts all upperase letter of given string to lowercase.
    text = text.lower()
    return text

# Code for removing repeated characters and punctuations
def reducing_incorrect_character_repeatation(text):
    """
    This Function will reduce repeatition to two characters 
    for alphabets and to one character for punctuations.    
    arguments:
         input_text: "text" of type "String".         
    return:
        value: Finally formatted text with alphabets repeating to 
        two characters & punctuations limited to one repeatition         
    Example:
    Input : Realllllllllyyyyy,        Greeeeaaaatttt   !!!!?....;;;;:)
    Output : Reallyy, Greeaatt !?.;:)    
    """
    # Pattern matching for all case alphabets
    Pattern_alpha = re.compile(r"([A-Za-z])\1{1,}", re.DOTALL)    
    # Limiting all the  repeatation to two characters.
    Formatted_text = Pattern_alpha.sub(r"\1\1", text)     
    # Pattern matching for all the punctuations that can occur
    Pattern_Punct = re.compile(r'([.,/#!$%^&*?;:{}=_`~()+-])\1{1,}')    
    # Limiting punctuations in previously formatted string to only one.
    Combined_Formatted = Pattern_Punct.sub(r'\1', Formatted_text)    
    # The below statement is replacing repeatation of spaces that occur more than two times with that of one occurrence.
    Final_Formatted = re.sub(' {2,}',' ', Combined_Formatted)
    return Final_Formatted

CONTRACTION_MAP = {
"ain't": "is not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he would",
"he'd've": "he would have",
"he'll": "he will",
"he'll've": "he he will have",
"he's": "he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how is",
"i'd": "i would",
"i'd've": "i would have",
"i'll": "i will",
"i'll've": "i will have",
"i'm": "i am",
"i've": "i have",
"isn't": "is not",
"it'd": "it would",
"it'd've": "it would have",
"it'll": "it will",
"it'll've": "it will have",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she would",
"she'd've": "she would have",
"she'll": "she will",
"she'll've": "she will have",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so as",
"that'd": "that would",
"that'd've": "that would have",
"that's": "that is",
"there'd": "there would",
"there'd've": "there would have",
"there's": "there is",
"they'd": "they would",
"they'd've": "they would have",
"they'll": "they will",
"they'll've": "they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we would",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what will",
"what'll've": "what will have",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"when's": "when is",
"when've": "when have",
"where'd": "where did",
"where's": "where is",
"where've": "where have",
"who'll": "who will",
"who'll've": "who will have",
"who's": "who is",
"who've": "who have",
"why's": "why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you would",
"you'd've": "you would have",
"you'll": "you will",
"you'll've": "you will have",
"you're": "you are",
"you've": "you have",
}
# The code for expanding contraction words
def expand_contractions(text, contraction_mapping =  CONTRACTION_MAP):
    """expand shortened words to the actual form.
       e.g. don't to do not    
       arguments:
            input_text: "text" of type "String".         
       return:
            value: Text with expanded form of shorthened words.        
       Example: 
       Input : ain't, aren't, can't, cause, can't've
       Output :  is not, are not, cannot, because, cannot have     
     """
    # Tokenizing text into tokens.
    list_Of_tokens = text.split(' ')

    # Checking for whether the given token matches with the Key & replacing word with key's value.
    
    # Check whether Word is in lidt_Of_tokens or not.
    for Word in list_Of_tokens: 
        # Check whether found word is in dictionary "Contraction Map" or not as a key. 
         if Word in CONTRACTION_MAP: 
                # If Word is present in both dictionary & list_Of_tokens, replace that word with the key value.
                list_Of_tokens = [item.replace(Word, CONTRACTION_MAP[Word]) for item in list_Of_tokens]
                
    # Converting list of tokens to String.
    String_Of_tokens = ' '.join(str(e) for e in list_Of_tokens) 
    return String_Of_tokens 

# The code for removing special characters
def removing_special_characters(text):
    """Removing all the special characters except the one that is passed within 
       the regex to match, as they have imp meaning in the text provided.    
    arguments:
         input_text: "text" of type "String".         
    return:
        value: Text with removed special characters that don't require.        
    Example: 
    Input : Hello, K-a-j-a-l. Thi*s is $100.05 : the payment that you will recieve! (Is this okay?) 
    Output :  Hello, Kajal. This is $100.05 : the payment that you will recieve! Is this okay?    
   """
    # The formatted text after removing not necessary punctuations.
    Formatted_Text = re.sub(r"[^a-zA-Z0-9:$-,%.?!]+", ' ', text) 
    # In the above regex expression,I am providing necessary set of punctuations that are frequent in this particular dataset.
    return Formatted_Text

#remove punctuations from a text
import re
def removing_punctuations(text):
    """Removing all the punctuations from the text.    
    arguments:
         input_text: "text" of type "String".         
    return:
        value: Text with removed punctuations that don't require.        
    Example: 
    Input:  I had such high hopes! for this dress size or (my usual size) to work for me.
    Output: I had such high hopes for this dress 15 size or my usual size to work for me    
   """
    #PUNCT_TO_REMOVE = string.punctuation
    #ans = text.translate(str.maketrans(", ", PUNCT_TO_REMOVE))
    ans = re.sub(r'[^\w\s]', '', text)
    return ans

# The code for removing stopwords
stoplist = stopwords.words('english') 
stoplist = set(stoplist)
def removing_stopwords(text):
    """This function will remove stopwords which doesn't add much meaning to a sentence 
       & they can be remove safely without comprimising meaning of the sentence.    
    arguments:
         input_text: "text" of type "String".         
    return:
        value: Text after omitted all stopwords.        
    Example: 
    Input : This is Kajal from delhi who came here to study.
    Output : ["'This", 'Kajal', 'delhi', 'came', 'study', '.', "'"]     
   """
    # repr() function actually gives the precise information about the string
    text = repr(text)
    # Text without stopwords
    No_StopWords = [word for word in word_tokenize(text) if word.lower() not in stoplist ]
    # Convert list of tokens_without_stopwords to String type.
    words_string = ' '.join(No_StopWords)    
    return words_string

# The code for spelling corrections
def spelling_correction(text):
    ''' 
    This function will correct spellings.    
    arguments:
         input_text: "text" of type "String".         
    return:
        value: Text after corrected spellings.        
    Example: 
    Input : This is Oberois from Dlhi who came heree to studdy.
    Output : This is Oberoi from Delhi who came here to study.    
    '''
    # Check for spellings in English language
    spell = Speller(lang='en')
    Corrected_text = spell(text)
    return Corrected_text

# The code for lemmatization
w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()
def lemmatization(text):
    """This function converts word to their root words 
       without explicitely cut down as done in stemming.    
    arguments:
         input_text: "text" of type "String".         
    return:
        value: Text having root words only, no tense form, no plural forms        
    Example: 
    Input : text reduced 
    Output :  text reduce    
   """
    # Converting words to their root forms
    lemma = [lemmatizer.lemmatize(w,'v') for w in w_tokenizer.tokenize(text)]
    return lemma

# Writing main function to merge all the preprocessing steps.
def text_preprocessing(text, accented_chars=True, contractions=True, lemma = True,
                        extra_whitespace=True, newlines_tabs=True, repeatition=True, 
                       lowercase=False, punctuations=False, mis_spell=False,
                       remove_html=True, links=True,  special_chars=True,
                       stop_words=False):
    """
    This function will preprocess input text and return
    the clean text.
    """        
    if newlines_tabs == True: #remove newlines & tabs.
        Data = remove_newlines_tabs(text)
    if remove_html == True: #remove html tags
        Data = strip_html_tags(Data)
    if links == True: #remove links
        Data = remove_links(Data)
    if extra_whitespace == True: #remove extra whitespaces
        Data = remove_whitespace(Data)
    if accented_chars == True: #remove accented characters
        Data = accented_characters_removal(Data)
    if lowercase == True: #convert all characters to lowercase
        Data = lower_casing_text(Data)
    if repeatition == True: #Reduce repeatitions   
        Data = reducing_incorrect_character_repeatation(Data)
    if contractions == True: #expand contractions
        Data = expand_contractions(Data)
    if special_chars == True: #remove special_chars
        Data = removing_special_characters(Data)
    if punctuations == True: #remove punctuations
        Data = removing_punctuations(Data)
    if stop_words == True: #Remove stopwords
        Data = removing_stopwords(Data)
    if mis_spell == True: #Check for mis-spelled words & correct them.
        Data = spelling_correction(Data)   
    if lemma == True: #Converts words to lemma form.
        Data = lemmatization(Data)           
    return Data

# #example code to load fake news data (CSV) and perform text processing and save the result to a CSV file
# # Read Dataset
# Df = pd.read_csv(r'../data/fakereal-news/New Task.csv', encoding = 'latin-1')
# print('Number of Data points : ', Df.shape[0])
# print('Number of features :', Df.shape[1])
# print('features :', Df.columns.values)
# # Show Dataset
# Df.drop(Df.columns[Df.columns.str.contains('Unnamed: 0',case = False)],axis = 1, inplace = True)
# Df.head()

# # This command tells information about the non-null values of attributes of Dataset.
# Df.info()

# Df['News_Headline'][0]

# # Shows statistics for every numerical column in our dataset.
# Df.describe()

# #Type of attribute "Title"
# type(Df['News_Headline'])

# # Pre-processing for Content
# List_Content = Df['News_Headline'].to_list()
# Final_Article = []
# Complete_Content = []
# for article in List_Content:
#     Processed_Content = text_preprocessing(article) #Cleaned text of Content attribute after pre-processing
#     Final_Article.append(Processed_Content)
# Complete_Content.extend(Final_Article)
# Df['Processed_Title'] = Complete_Content

# Df['Processed_Title']
# Df.head()

# Cleaned_Data = Df.to_csv('Cleaned_Data_with_Stopwords.csv', index = False)