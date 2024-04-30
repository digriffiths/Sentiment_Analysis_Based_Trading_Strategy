import re
from contractions import fix
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
import stanza

sns.set(style="darkgrid")

class SentimentAnalyser:
    def __init__(self):
        self.vader_analyzer = SentimentIntensityAnalyzer()
        self.stemmer = SnowballStemmer("english")
        self.stop_words = set([word for word in stopwords.words('english') if word != 'not'])
        self.stanza = stanza.Pipeline(lang='en', processors='tokenize,sentiment')

    def get_textblob_sentiment_score(self, text):
        analysis = TextBlob(text)
        return analysis.sentiment.polarity
    
    def get_stanfordnlp_score(self, text):
        doc = self.stanza(text)
        return doc.sentences[0].sentiment
    
    def get_vader_sentiment_score(self, text):
        scores = self.vader_analyzer.polarity_scores(text)
        return scores['compound']

    def get_sentiment_score(self, data):
        data['title_normalized'] = data['title'].apply(self.normalize_text)
        data['vader_sentiment_score'] = data['title_normalized'].apply(self.get_vader_sentiment_score)
        data['textblob_sentiment_score'] = data['title_normalized'].apply(self.get_textblob_sentiment_score)
        # data['stanfordnlp_score'] = data['title_normalized'].apply(self.get_stanfordnlp_score)
        return data

    def remove_non_english_characters(self, text):
        return re.sub(r'[^\x00-\x7F]+', "", text)

    def remove_numbers(self, text):
        return re.sub(r"\d+", "", text)

    def stem_tokens(self, tokens):
        return [self.stemmer.stem(token) for token in tokens]

    def remove_stop_words(self, tokens):
        return [token for token in tokens if token not in self.stop_words]

    def normalize_text(self, article):
        article = self.remove_non_english_characters(article)
        article = self.remove_numbers(article)
        article = fix(article)
        
        tokens = word_tokenize(article)
        tokens = self.stem_tokens(tokens)
        tokens = self.remove_stop_words(tokens)
        
        article = " ".join(tokens)
        return article
    
    def add_market_data(self, data, interval, ticker):
        start_date = data.index.min()
        end_date = data.index.max()
        data_resampled = data[['vader_sentiment_score', 'textblob_sentiment_score']].resample(interval).sum()
        market_data = self.get_market_data(ticker, start_date, end_date, interval)
        data_combined = data_resampled.join(market_data)
        return data_combined

    def run_strategy(self, data):
        for analyser in ['vader', 'textblob']:
        # window_list = [2, 5, 8, 10]
        # for window in window_list:
        #     data[f'ma_{window}'] = data['sentiment_score'].rolling(window).mean()
            data[f'{analyser}_position'] = np.where(data[f'{analyser}_sentiment_score'] > 0, 1, np.nan)
            data[f'{analyser}_position'] = np.where(data[f'{analyser}_sentiment_score'] == 0, 0, data[f'{analyser}_position'])
            data[f'{analyser}_position'] = np.where(data[f'{analyser}_sentiment_score'] < 0, -1, data[f'{analyser}_position'])
            data[f'{analyser}_position'] = data[f'{analyser}_position'].ffill()
            data[f'{analyser}_strategy'] = data['returns'] * data[f'{analyser}_position'].shift(1)
            data[f'{analyser}_cstrategy'] = data[f'{analyser}_strategy'].cumsum().apply(np.exp)
            data[f'{analyser}_long_price'] = data.creturns[data[f'{analyser}_position'].diff() == 2]
            data[f'{analyser}_short_price'] = data.creturns[data[f'{analyser}_position'].diff() == -2]
        return data

    def get_market_data(self, ticker, start_date, end_date, interval):
        data = yf.download(ticker, start=start_date, end=end_date, interval=interval, group_by='ticker', ignore_tz = False)
        data["returns"] = np.log(data.Close / data.Close.shift(1))
        data['creturns'] = data['returns'].cumsum().apply(np.exp)
        return data[["returns", "creturns"]]

    def plot_strat(self, df):
        fig, ax1 = plt.subplots(figsize=(12,8))
        title = "Sentiment Analysis Trading Strategy for Bitcoin News Headlines"

        creturns = ax1.plot(df.creturns, color='black', linewidth=1, label='BTC')
        cstrategy = ax1.plot(df['vader_cstrategy'], color='teal', linewidth=1, label='Vader Sentiment Analyser')
        cstrategy = ax1.plot(df['textblob_cstrategy'], color='purple', linewidth=1, label='Textblob Sentiment Analyser')

        # buy_signal = ax1.scatter(df.index , df['long_price'] , label = 'Long' , marker = '^', color = 'green',alpha =1 )
        # sell_signal = ax1.scatter(df.index , df['short_price'] , label = 'Short' , marker = 'v', color = 'red',alpha =1 )

        ax1.set_xlabel('date')
        ax1.set_ylabel('Returns')
        ax1.title.set_text(title)

        ax1.legend()
        plt.show()
