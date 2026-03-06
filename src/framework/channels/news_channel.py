from ghoshell_moss import PyChannel
from newsapi import NewsApiClient
from pydantic import BaseModel, Field
import trafilatura


class NewsSource(BaseModel):
    id: str|None = Field(default="")
    name: str = Field(default="")


class NewsDetails(BaseModel):
    source: NewsSource = Field(default_factory=NewsSource)
    author: str|None = Field(default="")
    title: str|None = Field(default="")
    description: str = Field(default="")
    url: str = Field(default="")
    urlToImage: str = Field(default="")
    publishedAt: str = Field(default="")
    content: str = Field(default="")

    fullContent: str = Field(default="")


class NewsResponse(BaseModel):
    status: str = Field(default="")
    total: str = Field(default="")
    news: list[NewsDetails] = Field(default=[])


class NewsAPI:
    def __init__(self, api_key: str):
        self.newsapi = NewsApiClient(api_key=api_key)

    async def get_news(self, q: str, page_size: int = 10, page: int = 1):
        """
        获取新闻
        :param q: 新闻关键词
        :param page_size: 每页新闻数量
        :param page: 新闻页码
        :return:
        """
        res = self.newsapi.get_everything(q=q, page_size=page_size, page=page)

        status = res["status"]
        total = res["totalResults"]
        articles = res["articles"]

        news = []
        for item in articles:
            details = NewsDetails.model_validate(item)
            downloaded = trafilatura.fetch_url(details.url)
            details.fullContent = trafilatura.extract(downloaded, output_format="markdown")
            news.append(details)

        return NewsResponse(
            status=status,
            total=total,
            news=news
        )

    def as_channel(self):
        chan = PyChannel(name="news", block=True)
        chan.build.command()(self.get_news)
        return chan


async def main():
    newsapi = NewsAPI(api_key="1d8d11c1d95c415a9856c1bc05836076")
    news_response = await newsapi.get_news(q="伊朗最新局势")
    print(news_response)

if __name__ == "__main__":
    import os
    # 基于Shadowsocks的代理，翻墙
    os.environ["http_proxy"] = "http://127.0.0.1:1087"
    os.environ["https_proxy"] = "http://127.0.0.1:1087"
    import asyncio
    asyncio.run(main())
