from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import glob, os
import re

def sentiment_analysis(df, filename):
    # apply sentiment analysis
    analyser = SentimentIntensityAnalyzer()

    sentiment_score_list = []
    sentiment_scores_list = []
    sentiment_label_list = []

    for i in df['processed_text'].values.tolist():
        tmp = {}
        sentiment_score = analyser.polarity_scores(i)

        if sentiment_score['compound'] >= 0.05:
            sentiment_score_list.append(sentiment_score['compound'])
            sentiment_label_list.append('positive')
        elif sentiment_score['compound'] <= -0.05:
            sentiment_score_list.append(sentiment_score['compound'])
            sentiment_label_list.append('negative')
        else:
            sentiment_score_list.append(sentiment_score['compound'])
            sentiment_label_list.append('neutral')



        
    df['sentiment'] = sentiment_label_list
    df['sentiment_score'] = sentiment_score_list

    df['topic'] = df['topic'].str.replace("'", "", regex=False)
    df['topic'] = df.topic.apply(lambda x: x[1:-1].split(', '))
    df = df.explode('topic')

    df_mean = df.groupby(['topic'])[['sentiment_score']].median().reset_index()

    tmp = []
    for i in df_mean[['topic', 'sentiment_score']].values.tolist():
        tmp_1 = []
        if i[1] >= 0.05:
            tmp_1.append(i[0])
            tmp_1.append('positive')
            tmp_1.append(i[1])
        elif i[1] <= -0.05:
            tmp_1.append(i[0])
            tmp_1.append('negative')
            tmp_1.append(i[1])
        else:
            tmp_1.append(i[0])
            tmp_1.append('neutral')
            tmp_1.append(i[1])
        tmp.append(tmp_1)
    
    df = pd.DataFrame(tmp, columns=['topic', 'sentiment', 'sentiment_score'])

    df.to_csv('PAPER/{}.csv'.format(filename), index=False)


if __name__ == '__main__':
    files = glob.glob("PAPER/*.csv")

    for i in files:
        df = pd.read_csv(i, lineterminator='\n')
        sentiment_analysis(df, i[9:-4])