# News Sentiment Analysis For Algorithmic Trading

## Description

This project explores the application of sentiment analysis in developing a trading strategy. It employs 2 sentiment analysis tools, VADER and TextBlob, to derive sentiment scores from Bitcoin news articles and uses these scores to inform trading decisions.

## Why?

The motivation behind this project is to leverage the predictive power of sentiment analysis to enhance trading strategies. By analyzing the sentiment from Bitcoin news articles, the project aims to identify potential market movements and improve trading decisions, ultimately aiming to increase profitability and reduce risk.

## Data

The dataset includes Bitcoin news articles from January 1, 2020, to December 31, 2022, featuring timestamps and article titles. The titles are normalized through a series of preprocessing steps to enhance the accuracy of sentiment analysis. The normalized titles are then analyzed using VADER and TextBlob to obtain sentiment scores.

## Sentiment Analysis Tools

- **VADER**: Optimized for social media texts, VADER provides sentiment scores using a human-curated lexicon, effective across various domains without requiring training data.
- **TextBlob**: Offers a straightforward API for common NLP tasks, ideal for rapid sentiment analysis evaluations.

## Features

- **Sentiment Scoring**: Calculates sentiment scores for Bitcoin news headlines using VADER and TextBlob.
- **Market Data Integration**: Enhances sentiment data with Bitcoin market data including price and returns.
- **Data Resampling**: Transforms high-frequency sentiment and market data into a daily frequency for analysis.
- **Strategy Execution**: Deploys a trading strategy based on sentiment and market data.
- **Visualization**: Graphically represents the trading strategy's performance.

## Usage

1. **Data Preparation**: Import and preprocess the Bitcoin news data.
2. **Sentiment Scoring**: Conduct sentiment analysis on the cleaned data.
3. **Market Data Integration**: Merge Bitcoin market data with sentiment scores.
4. **Data Resampling**: Aggregate the combined data to a daily frequency.
5. **Run Strategy**: Implement the trading strategy using the processed data.
6. **Plot Results**: Display the strategy's performance over time through visualizations.

## Contributing

Contributions to this project are welcome! To contribute, please fork the repository, create a new branch for your contributions, and submit a pull request for review.
