import requests
from urllib.parse import urlencode
import json
import feedparser

words = ['primary',
         'convention',
         'hype',
         'immigration',
         'big',
         'amid',
         'change',
         'campaign',
         'he',
         'nomination',
         'says',
         'endorsement',
         'women',
         'police',
         'critics',
         'state',
         'want',
         'conservatives',
         'poll',
         'himself',
         'coal',
         'millennials',
         'would',
         'see',
         'prices',
         'wage',
         'protests',
         'battle',
         'moves',
         'claims']

titles = {}
for word in words:
    titles[word] = []
    query = urlencode({'q': f'Searchterm "{word}"'})
    url = "https://news.google.com/rss/search?" + query

    feeds = feedparser.parse(url).entries
    print("Number of articles", len(feeds))
    counter = 0
    for f in feeds:
        title = f['title']
        if word.lower() in title.lower().split():
            titles[word].append(f['title'])
            counter += 1
    print("Number of articles scrapped", counter)

for word, title_list in titles.items():
    print(f"============== {word} ===============")
    for title in title_list:
        print(title)

with open("/Users/antonioloison/Projects/fakenewsdetec/data/titles_3.json", "w") as f:
    json.dump(titles, f)
