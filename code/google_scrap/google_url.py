try:
    import csv
    import requests
    import pandas as pd
    from itertools import chain
    from newspaper import Article
    from bs4 import BeautifulSoup
    from googlesearch import search
except ImportError:
    print("No module named 'google' found")

# to search
# 'US Election 2024', 'US Election 2024 in New York Times', 'US Election 2024 in Washington post',
#            'US Election 2024 on CNN', 'US Election 2024 on Fox News Channel', 'US Election 2024 on ABC News',
#            'US Election 2024 on CBS News', 'US Election 2024 on MSNBC', 'US Election 2024 on NBC News',
#            'US Election 2024 on USA Today', 'US Election 2024 in The Wall Street Journal',
#            'US Election 2024 in POLITICO', 'US Election 2024 in Bloomberg', 'US Election 2024 in Vice News',
#            'US Election 2024 in HuffPost',

queries = ['US Election 2024 in  in Time Magazine',
           'US Election 2024 in U.S. News & World Report Magazine', 'US Election 2024 in in Newsweek Magazine']

count = 0

# for query in queries:
#     result = []
#     print('Starting for = ', query)
#     for j in search(query, pause=10.0):
#         result.append(j)
#
#     result = pd.DataFrame(result, columns=['URL'])
#     result.to_csv('../../data/url_us_election_2024.csv', mode='a', index=False)
#     print('Done for = ', query)

links = pd.read_csv('../../data/url_us_election_2024.csv').values.tolist()

# Get unique list
links = list(chain.from_iterable(links))
links = pd.DataFrame(set(links), columns=['URL'])


for link in links['URL']:
    try:
        url_article = Article(link, language="en") # en for English
        url_article.download()

        # To parse the article
        url_article.parse()

        # To perform natural language processing ie..nlp
        url_article.nlp()
    except:
        continue
    # To extract title
    title = pd.DataFrame([url_article.title], columns=['Title'])

    # To extract text
    text = pd.DataFrame([url_article.text], columns=['Text'])

    # To extract summary
    summary = pd.DataFrame([url_article.summary], columns=['Summary'])

    # To extract keywords
    keywords = pd.DataFrame([', '.join(url_article.keywords)], columns=['Keywords'])

    # Source URL
    link = pd.DataFrame([link], columns=['Link'])

    result = pd.concat([title, text, summary, keywords, link], axis=1)
    print(result)
    result.to_csv('../../data/extracted_data_us_election_2024.csv', mode='a', index=False, header=False)
