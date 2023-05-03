try:
    from googlesearch import search
    import pandas as pd
except ImportError:
    print("No module named 'google' found")

# to search
queries = ['US Election 2024', 'US Election 2024 in New York Times', 'US Election 2024 in Washington post',
           'US Election 2024 on CNN', 'US Election 2024 on Fox News Channel', 'US Election 2024 on ABC News',
           'US Election 2024 on CBS News', 'US Election 2024 on MSNBC', 'US Election 2024 on NBC News',
           'US Election 2024 on USA Today', 'US Election 2024 in The Wall Street Journal',
           'US Election 2024 in POLITICO', 'US Election 2024 in Bloomberg', 'US Election 2024 in Vice News',
           'US Election 2024 in HuffPost', 'US Election 2024 in  in Time Magazine',
           'US Election 2024 in U.S. News & World Report Magazine', 'US Election 2024 in in Newsweek Magazine']

count = 0

for query in queries:
    result = []
    for j in search(query, pause=10.0, ):
        result.append(j)

    result = pd.DataFrame(result, columns=['URL'])
    result.to_csv('../../data/url_us_election_2024.csv', mode='a', index=False)
    print('Done for = ', query)