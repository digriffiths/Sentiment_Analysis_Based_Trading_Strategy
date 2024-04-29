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
sns.set(style="darkgrid")

class SentimentAnalyser:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
        self.stemmer = SnowballStemmer("english")
        self.stop_words = set([word for word in stopwords.words('english') if word != 'not'])

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

    def get_score(self, text):
        scores = self.analyzer.polarity_scores(text)
        return scores['compound']
    
    def get_sentiment_score(self, data):
        data['title_normalized'] = data['title'].apply(self.normalize_text)
        data['sentiment_score'] = data['title_normalized'].apply(self.get_score)
        return data
    
    def add_market_data(self, data, interval, ticker):
        start_date = data.index.min()
        end_date = data.index.max()
        data_resampled = data[['sentiment_score']].resample(interval).sum()
        market_data = self.get_market_data(ticker, start_date, end_date, interval)
        data_combined = data_resampled.join(market_data)
        return data_combined

    def run_strategy(self, data):
        # window_list = [2, 5, 8, 10]
        # for window in window_list:
        #     data[f'ma_{window}'] = data['sentiment_score'].rolling(window).mean()
        data['position'] = np.where(data['sentiment_score'] > 0, 1, np.nan)
        data['position'] = np.where(data['sentiment_score'] == 0, 0, data['position'])
        data['position'] = np.where(data['sentiment_score'] < 0, -1, data['position'])
        data['position'] = data['position'].ffill()
        data['strategy'] = data['returns'] * data['position'].shift(1)
        data['creturns'] = data["returns"].cumsum().apply(np.exp)
        data['cstrategy'] = data["strategy"].cumsum().apply(np.exp)
        data["long_price"] = data.creturns[data['position'].diff() == 2]
        data["short_price"] = data.creturns[data['position'].diff() == -2]
        return data

    def get_market_data(self, ticker, start_date, end_date, interval):
        data = yf.download(ticker, start=start_date, end=end_date, interval=interval, group_by='ticker', ignore_tz = False)
        data["returns"] = np.log(data.Close / data.Close.shift(1))
        return data[["returns"]]

    def plot_strat(self, df):
        fig, ax1 = plt.subplots(figsize=(12,8))
        title = "Sentiment Analysis Trading Strategy for Bitcoin News Headlines"

        creturns = ax1.plot(df.creturns, color='black', linewidth=1, label='BTC')
        cstrategy = ax1.plot(df['cstrategy'], color='teal', linewidth=1, label='Strategy')
        buy_signal = ax1.scatter(df.index , df['long_price'] , label = 'Long' , marker = '^', color = 'green',alpha =1 )
        sell_signal = ax1.scatter(df.index , df['short_price'] , label = 'Short' , marker = 'v', color = 'red',alpha =1 )

        ax1.set_xlabel('date')
        ax1.set_ylabel('Returns')
        ax1.title.set_text(title)

        ax1.legend()
        plt.show()