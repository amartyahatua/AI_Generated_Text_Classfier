import wikipediaapi
import pandas as pd
import os
import time

# Source data: https://huggingface.co/datasets/aadityaubhat/GPT-wiki-intro/blob/main/GPT-wiki-intro.csv.zip
gpt_wiki_data = pd.read_csv('../../data/GPT-wiki-intro.csv')

# Using the titles to extract the data
gpt_wiki_title = gpt_wiki_data['title']

result = []

'''
This is a recursive function, which extracts the data from wiki pages and saves in a list. 
title: Title of the page
sections: Sections of a page
'''

def wiki_sections(title, sections, level=0):
    for s in sections:
        temp = []
        temp.append(title)
        temp.append(s.title)
        temp.append(s.text)
        result.append(temp)
        wiki_sections(title, s.sections, level + 1)


wiki_wiki = wikipediaapi.Wikipedia(
    language='en',
    extract_format=wikipediaapi.ExtractFormat.WIKI
)

# Downloading fist 100 wiki links
dest_file = '../../data/wiki_data_extract_1_50000.csv'
if os. path. isfile(dest_file):
    prev_df = pd.read_csv(dest_file)
    prev_count = prev_df.shape[0]
else:
    prev_count = 0

count = 0
for title in gpt_wiki_title:
    count += 1
    if(prev_count < count):
        if(count %500 == 1):
            print('Done: ',count)
            print("Sleeping...")
            time.sleep(30)
            print('Working...')

        if(count < 50000):
            page_py = wiki_wiki.page(title)
            #print("Page - Title: %s" % page_py.title)
            wiki_sections(page_py.title, page_py.sections)
        else:
            result = pd.DataFrame(result, columns=['Page title', 'Section title', 'Text'])
            result.to_csv('../../data/wiki_data_extract_1_50000.csv', mode='a', index=False, header=False)
            result = []
            break
        # Saving the result as .csv file
        result = pd.DataFrame(result, columns=['Page title', 'Section title', 'Text'])
        result.to_csv('../../data/wiki_data_extract_1_50000.csv', mode='a', index=False, header=False)
        result = []

prev_df = pd.read_csv(dest_file)
print(prev_df.shape)