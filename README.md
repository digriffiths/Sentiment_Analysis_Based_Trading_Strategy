# Sentiment Analysis Based Trading Strategy

## Overview

This project explores the application of sentiment analysis in developing a trading strategy. It generates sentiment scores using various analyzers to inform trading decisions.

## Data

The data consists of Bitcoin news articles from January 1, 2020, to December 31, 2022. The dataset includes timestamps and article titles. The titles are normalized and processed to extract sentiment scores.

## Sentiment Analysis

We utilize the VADER (Valence Aware Dictionary and sEntiment Reasoner) and TextBlob sentiment analysis tools. VADER is specifically tuned for social media texts but works well across various domains, providing a sentiment score without the need for training data, using a human-curated sentiment lexicon. TextBlob, on the other hand, offers a simple API for common natural language processing (NLP) tasks and is useful for quick sentiment analysis evaluations.

## Features

- **Sentiment Scoring**: Assigns sentiment scores to Bitcoin news headlines using VADER and Textblob.
- **Market Data Integration**: Adds Bitcoin market data (price, returns) to the sentiment data.
- **Resampling**: Converts the high-frequency sentiment and market data into daily frequency for analysis.
- **Strategy Execution**: Implements a trading strategy based on the sentiment and market data.
- **Visualization**: Plots the performance of the trading strategy.

## Usage

1. **Data Preparation**: Load and clean the Bitcoin news data.
2. **Sentiment Scoring**: Apply the sentiment analysis on the cleaned data.
3. **Market Data Addition**: Integrate Bitcoin market data.
4. **Resample Data**: Aggregate the data to daily frequency.
5. **Run Strategy**: Execute the trading strategy based on the processed data.
6. **Plot Results**: Visualize the strategy performance over time.

## Installation

Ensure you have Python installed and then set up a virtual environment:
