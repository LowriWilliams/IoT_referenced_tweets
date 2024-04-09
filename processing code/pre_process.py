import nltk
from nltk.corpus import stopwords
from stop_words_list import stop_words_list
from nltk.stem import PorterStemmer
import re
import pandas as pd
import glob, os
from dateutil.parser import parse
from nltk.util import ngrams



# nltk.download('stopwords')
# nltk.download('punkt')

# initialise stemmer
stemmer = PorterStemmer()

def pre_process(df, i):

    # initiate stopwords from nltk
    stop_words = stopwords.words('english')

    # add additional missing terms
    stop_words.extend(stop_words_list) 

    RE_EMOJI = re.compile('[\U00010000-\U0010ffff]', flags=re.UNICODE)

    df['processed_text'] = df['tweet']

    remove = ["&lt;", "&gt;", "&le;", "&ge;", "&amp;", "‚Äô", "\n", "@[A-Za-z0-9_]+", 'https?:\/\/\S*', "#[A-Za-z0-9_]+", RE_EMOJI, "RT ", "_"]

    for r in remove:
        df['processed_text'] = df['processed_text'].str.replace(r, "  ", regex=True)

    df['processed_text'] = df['processed_text'].str.replace('[^\x00-\x7F]+', ' ', regex=True)

    # remove punct
    df['processed_text'] = df['processed_text'].str.replace('[^\w\s]', ' ', regex=True).str.replace(' +', ' ', regex=True)
    df['processed_text'] = df['processed_text'].str.strip().str.lower()

    # remove the search term from the text itself
    search_term = ['iot', 'internet of things']
    
    df['processed_text'] = [' '.join([y for y in x.split() if y not in search_term]) for x in df['processed_text']]

    df = df.drop_duplicates(subset=['processed_text'], keep=False)

    # tokenise string
    df = df.copy()
    df['processed_text_tokens'] = df.apply(lambda row: nltk.word_tokenize(row['processed_text']), axis=1)

    # remove just numbers
    tmp = []
    for d in df['processed_text_tokens'].values.tolist(): 
        tmp.append([x for x in d if not (x.isdigit() or x[0] == '-' and x[1:].isdigit())])
    
    df['processed_text_tokens'] = tmp

    # remove stopwords
    df['processed_text_tokens'] = df['processed_text_tokens'].apply(lambda x: [item for item in x if item not in stop_words])

    # stem words
    # df['processed_text_tokens'] = df['processed_text_tokens'].apply(lambda x: [stemmer.stem(y) for y in x]) 

    # add bigrams to list
    tmp = []
    for t in df['processed_text_tokens'].values.tolist():
        bigram = list(ngrams(t, 2))
        for b in bigram:
            t.append(" ".join(b))
        tmp.append(t)
    
    df['processed_text_tokens'] = tmp
    
    df = df[df['processed_text_tokens'].map(lambda d: len(d)) > 0]
    
    df.to_csv('pre_processed/{}.csv'.format(i[5:-4]), index=False)


files = glob.glob("data/*.csv")

for i in files:
    df = pd.read_csv(i, lineterminator='\n', low_memory=False)
    pre_process(df, i)