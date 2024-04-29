import requests
from bs4 import BeautifulSoup
import pandas as pd

class NewsRetriever:
    """
    A class used to scrape news articles from Google News based on specific queries and date ranges.
    """
    def __init__(self, base_url:str):
        """
        Initializes the NewsRetriever with the base URL for Google News.
        """
        self.base_url = base_url

    def fetch_news(self, query: str, date_ranges: list[tuple[str, str]]) -> pd.DataFrame:
        """
        Fetches news articles based on a search query and a list of date ranges.

        Args:
            query (str): The search query for fetching news.
            date_ranges (list of tuples): A list of tuples where each tuple contains start and end dates.

        Returns:
            pandas.DataFrame: A DataFrame containing the titles, dates, and links of the articles, indexed by date.
        """
        self.query = query
        self.date_ranges = date_ranges
        newslist = []
        for start, end in date_ranges:
            url = f'{self.base_url}/search?q={query}%20before%3A{end}%20after%3A{start}&hl=en-GB&gl=GB&ceid=GB%3Aen'
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            articles = soup.find_all('article')

            for article in articles:
                title, date = self.extract_article_details(article)
                if title and date:
                    newslist.append((title, date))

        return pd.DataFrame(newslist, columns=['title', 'date']).set_index('date')

    def extract_article_details(self, article: BeautifulSoup) -> tuple[str, str, pd.Timestamp]:
        """
        Extracts details from a single article element.

        Args:
            article (bs4.element.Tag): A BeautifulSoup Tag object representing an article.

        Returns:
            tuple: A tuple containing the title, link, and date of the article.
        """
        title_element = article.find('a', class_='JtKRv')
        title = title_element.get_text(strip=True) if title_element else None

        date_element = article.find('time', class_='hvbAAd')
        date = pd.to_datetime(date_element.get('datetime')) if date_element else None

        return title, date

    def save_to_csv(self, df: pd.DataFrame) -> None:
        """
        Saves the DataFrame to a CSV file.

        Args:
            df (pandas.DataFrame): The DataFrame to save.
        """
        csv_filename = f'news_{self.query}_{self.date_ranges[0][0]}_{self.date_ranges[-1][1]}.csv'
        df.to_csv(csv_filename)
        print(f"Data saved to {csv_filename}")

