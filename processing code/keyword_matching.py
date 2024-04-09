import pandas as pd
import re
import glob


def matching(terms, df, filename):
    tmp_k = []
    for i, row in df.iterrows():
        tmp = []
        row['processed_text_tokens'] = row['processed_text_tokens'].replace("[", "").replace("]", "").replace("'", "")
        row['processed_text_tokens'] = row['processed_text_tokens'].split(', ')

        for k in row['processed_text_tokens']:
            if k in terms:
                if k not in tmp:
                    tmp.append(k)
        tmp_k.append(tmp)

    df['topic'] = tmp_k
    df = df[df['topic'].map(lambda d: len(d)) > 0]
    df.to_csv('keywords/{}.csv'.format(filename), index=False)


if __name__ == '__main__':
    cybok = pd.read_csv('cybok.csv')

    cybok['term'] = cybok['term'].str.split(' - ')

    terms = []

    for i in cybok['term'].values.tolist():
        for j in i:
            if j.lower() not in terms:
                j = j.lower()
                terms.append(re.sub('[^\w\s]', ' ', j))

    terms = [re.sub('\s+', ' ', x) for x in terms]

    files = glob.glob("pre_processed/*.csv")

    for i in files:
        df = pd.read_csv(i, lineterminator='\n')
        matching(terms, df, i[14:-4])