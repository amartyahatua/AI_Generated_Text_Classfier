try:
	from googlesearch import search
	import pandas as pd
except ImportError:
	print("No module named 'google' found")

# to search
query = "US Election 2024"

count = 0
result = []
for j in search(query):
	count += 1
	result.append(j)
	print('Number of URL got = ', count)

result = pd.DataFrame(result, columns=['URL'])
result.to_csv('../../data/url_us_election_2024.csv',index=False)

