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
import pandas as pd

sns.set(style="darkgrid")

class SentimentAnalyser:
    def __init__(self) -> None:
        """
        Initializes the SentimentAnalyser with various sentiment analysis tools and preprocessing utilities.
        """
        self.vader_analyzer = SentimentIntensityAnalyzer()
        self.stemmer = SnowballStemmer("english")
        self.stop_words = set([word for word in stopwords.words('english') if word != 'not'])
        self.stanza = stanza.Pipeline(lang='en', processors='tokenize,sentiment')

    def get_textblob_sentiment_score(self, text: str) -> float:
        """
        Analyzes the sentiment of the given text using TextBlob.

        Args:
            text (str): The text to analyze.

        Returns:
            float: The polarity score of the text.
        """
        analysis = TextBlob(text)
        return analysis.sentiment.polarity
    
    def get_stanfordnlp_score(self, text: str) -> int:
        """
        Analyzes the sentiment of the given text using StanfordNLP.

        Args:
            text (str): The text to analyze.

        Returns:
            int: The sentiment score of the text.
        """
        doc = self.stanza(text)
        return doc.sentences[0].sentiment
    
    def get_vader_sentiment_score(self, text: str) -> float:
        """
        Analyzes the sentiment of the given text using VADER.

        Args:
            text (str): The text to analyze.

        Returns:
            float: The compound sentiment score of the text.
        """
        scores = self.vader_analyzer.polarity_scores(text)
        return scores['compound']

    def get_sentiment_score(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Computes sentiment scores for the titles in the given DataFrame.

        Args:
            data (pd.DataFrame): The DataFrame containing article titles.

        Returns:
            pd.DataFrame: The DataFrame with added sentiment scores.
        """
        data['title_normalized'] = data['title'].apply(self.normalize_text)
        data['vader_sentiment_score'] = data['title_normalized'].apply(self.get_vader_sentiment_score)
        data['textblob_sentiment_score'] = data['title_normalized'].apply(self.get_textblob_sentiment_score)
        # data['stanfordnlp_score'] = data['title_normalized'].apply(self.get_stanfordnlp_score)
        return data

    def remove_non_english_characters(self, text: str) -> str:
        """
        Removes non-English characters from the given text.

        Args:
            text (str): The text to process.

        Returns:
            str: The text with non-English characters removed.
        """
        return re.sub(r'[^\x00-\x7F]+', "", text)

    def remove_numbers(self, text: str) -> str:
        """
        Removes numbers from the given text.

        Args:
            text (str): The text to process.

        Returns:
            str: The text with numbers removed.
        """
        return re.sub(r"\d+", "", text)

    def stem_tokens(self, tokens: list[str]) -> list[str]:
        """
        Stems the given list of tokens.

        Args:
            tokens (list[str]): The list of tokens to stem.

        Returns:
            list[str]: The list of stemmed tokens.
        """
        return [self.stemmer.stem(token) for token in tokens]

    def remove_stop_words(self, tokens: list[str]) -> list[str]:
        """
        Removes stop words from the given list of tokens.

        Args:
            tokens (list[str]): The list of tokens to process.

        Returns:
            list[str]: The list of tokens with stop words removed.
        """
        return [token for token in tokens if token not in self.stop_words]

    def normalize_text(self, article: str) -> str:
        """
        Normalizes the given article text by removing non-English characters, numbers, expanding contractions, tokenizing, stemming, and removing stop words.

        Args:
            article (str): The article text to normalize.

        Returns:
            str: The normalized article text.
        """
        article = self.remove_non_english_characters(article)
        article = self.remove_numbers(article)
        article = fix(article)
        
        tokens = word_tokenize(article)
        tokens = self.stem_tokens(tokens)
        tokens = self.remove_stop_words(tokens)
        
        article = " ".join(tokens)
        return article
    
    def add_market_data(self, data: pd.DataFrame, interval: str, ticker: str) -> pd.DataFrame:
        """
        Adds market data to the given DataFrame based on the specified interval and ticker.

        Args:
            data (pd.DataFrame): The DataFrame containing sentiment scores.
            interval (str): The resampling interval for the market data.
            ticker (str): The ticker symbol for the market data.

        Returns:
            pd.DataFrame: The DataFrame with added market data.
        """
        start_date = data.index.min()
        end_date = data.index.max()
        data_resampled = data[['vader_sentiment_score', 'textblob_sentiment_score']].resample(interval).sum()
        market_data = self.get_market_data(ticker, start_date, end_date, interval)
        data_combined = data_resampled.join(market_data)
        return data_combined

    def run_strategy(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Runs the trading strategy based on sentiment scores and calculates positions and strategy returns.

        Args:
            data (pd.DataFrame): The DataFrame containing sentiment scores and market data.

        Returns:
            pd.DataFrame: The DataFrame with added strategy positions and returns.
        """
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

    def get_market_data(self, ticker: str, start_date: pd.Timestamp, end_date: pd.Timestamp, interval: str) -> pd.DataFrame:
        """
        Fetches market data for the given ticker and date range, and calculates returns and cumulative returns.

        Args:
            ticker (str): The ticker symbol for the market data.
            start_date (pd.Timestamp): The start date for the market data.
            end_date (pd.Timestamp): The end date for the market data.
            interval (str): The resampling interval for the market data.

        Returns:
            pd.DataFrame: The DataFrame containing market returns and cumulative returns.
        """
        data = yf.download(ticker, start=start_date, end=end_date, interval=interval, group_by='ticker', ignore_tz=False)
        data["returns"] = np.log(data.Close / data.Close.shift(1))
        data['creturns'] = data['returns'].cumsum().apply(np.exp)
        return data[["returns", "creturns"]]

    def plot_strat(self, df: pd.DataFrame) -> None:
        """
        Plots the trading strategy performance.

        Args:
            df (pd.DataFrame): The DataFrame containing strategy returns and cumulative returns.
        """
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
