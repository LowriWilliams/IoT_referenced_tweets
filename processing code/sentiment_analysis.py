from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import glob, os
import re

def sentiment_analysis(df, filename):
    # apply sentiment analysis
    analyser = SentimentIntensityAnalyzer()

    sentiment_score_list = []
    sentiment_label_list = []

    for i in df['processed_text'].values.tolist():
        sentiment_score = analyser.polarity_scores(i)

        if sentiment_score['compound'] >= 0.05:
            sentiment_score_list.append(sentiment_score['compound'])
            sentiment_label_list.append('positive')
        elif sentiment_score['compound'] > -0.05 and sentiment_score['compound'] < 0.05:
            sentiment_score_list.append(sentiment_score['compound'])
            sentiment_label_list.append('neutral')
        elif sentiment_score['compound'] <= -0.05:
            sentiment_score_list.append(sentiment_score['compound'])
            sentiment_label_list.append('negative')


        
    df['sentiment'] = sentiment_label_list
    df['sentiment_score'] = sentiment_score_list

    df['topic'] = df['topic'].str.replace("'", "", regex=False)
    df['topic'] = df.topic.apply(lambda x: x[1:-1].split(', '))
    df = df.explode('topic')

    df['sentiment_score'] = df['sentiment_score'].abs()
    df_mean = df.groupby(['topic', 'sentiment'])[['sentiment_score']].mean().reset_index()
    df_mean = df_mean.sort_values('sentiment_score').drop_duplicates(['topic'],keep='last')
    
    df_mean.to_csv('test_final/{}.csv'.format(filename), index=False)


if __name__ == '__main__':
    files = glob.glob("keywords/*.csv")

    for i in files:
        df = pd.read_csv(i, lineterminator='\n')
        sentiment_analysis(df, i[9:-4])