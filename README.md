# News Sentiment Analysis For Algorithmic Trading

## Description

This project focuses on leveraging news article sentiments to inform trading strategies. It starts by using a custom-built tool to scrape news articles, extracting titles and dates, which are then stored in a pandas DataFrame. The sentiment of each article is analyzed using VADER and TextBlob, two different sentiment analysis tools. These sentiments are used to generate trading signals that dictate whether to take a long, short, or neutral position based on the sentiment score. The trading strategy's performance, influenced by these signals, is visualized to compare the effectiveness of the sentiment analysis methods over time, providing insights into how news sentiment impacts market behavior.

## Why?

The project was initiated to investigate the impact of news on financial markets, specifically to explore if it is possible to generate trading signals based on news sentiment analysis. The promising results indicate the potential for developing a model suitable for live trading, providing valuable insights into how news sentiment correlates with market movements and affects trading outcomes.

## Data

The dataset includes Bitcoin news articles from January 1, 2020, to December 31, 2022, featuring timestamps and article titles. The titles are normalized through a series of preprocessing steps to enhance the accuracy of sentiment analysis.

## Sentiment Analysis Tools

- **VADER**: Optimized for social media texts, VADER provides sentiment scores using a human-curated lexicon, effective across various domains without requiring training data.
- **TextBlob**: Offers a straightforward API for common NLP tasks, ideal for rapid sentiment analysis evaluations.

## Usage

To use this project for sentiment analysis and trading signal generation, follow these steps:

1. **Clone the Repository:**

   ```
   git clone https://github.com/digriffiths/Sentiment_Analysis_Based_Trading_Strategy.git
   cd news-sentiment-trading
   ```

2. **Install Dependencies:**
   Ensure you have Python installed, then run:

   ```
   pip install -r requirements.txt
   ```

3. **Run the get_news.ipynb to download news articles or use the csv provided:**
   Execute the scraper to collect the latest news articles:

   ```
   get_news.ipynb
   news_bitcoin_2020-01-01_2022-12-31.csv
   ```

4. **Run the sentiment_analyser.ipynb to analyse the sentiment of the news articles:**
   Analyze the sentiments of the scraped news articles using:

   ```
   sentiment_analyser.ipynb
   ```

For detailed documentation on each step, refer to the respective script's comments or the project's wiki.

## Contributing

Contributions to this project are welcome! To contribute, please fork the repository, create a new branch for your contributions, and submit a pull request for review.
