import threading
from typing import List

import requests
from ghoshell_container import Provider, IoCContainer, INSTANCE
from ghoshell_moss import PyChannel, Message, Text
from newsapi import NewsApiClient
from pydantic import BaseModel, Field
import trafilatura
import os

from framework.abcd.agent import EventBus
from framework.abcd.agent_event import ReactAgentEvent


trafilatura.PROXY_URL = "http://127.0.0.1:1087"


class NewsSource(BaseModel):
    id: str|None = Field(default="")
    name: str = Field(default="")


class NewsDetails(BaseModel):
    source: NewsSource = Field(default_factory=NewsSource)
    author: str|None = Field(default="")
    title: str|None = Field(default="")
    description: str|None = Field(default="")
    url: str|None = Field(default="")
    urlToImage: str|None = Field(default="")
    publishedAt: str|None = Field(default="")
    content: str|None = Field(default="")

    fullContent: str = Field(default="")


class NewsResponse(BaseModel):
    status: str = Field(default="")
    total: str = Field(default="")
    news: list[NewsDetails] = Field(default=[])


class NewsAPI:
    def __init__(self, api_key: str, eventbus: EventBus=None):
        self._eventbus = eventbus
        session = requests.Session()
        session.proxies = {
            "http": "http://127.0.0.1:1087",
            "https": "http://127.0.0.1:1087"
        }
        self.newsapi = NewsApiClient(api_key=api_key, session=session)

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

        news: List[NewsDetails] = []
        for item in articles:
            details = NewsDetails.model_validate(item)

            result = {"value": None, "exception": None}

            # 定义包装函数，用于在线程中执行目标函数
            def wrapper():
                try:
                    result["value"] = trafilatura.fetch_url(details.url)
                except Exception as e:
                    result["exception"] = e

            thread = threading.Thread(target=wrapper)
            thread.daemon = True
            thread.start()
            thread.join(timeout=2)

            if result["value"]:
                # downloaded = trafilatura.fetch_url(details.url)
                details.fullContent = trafilatura.extract(result["value"], output_format="markdown")
                news.append(details)

        if self._eventbus:
            await self._eventbus.put(ReactAgentEvent(
                messages=[Message.new(role="system").with_content(
                    Text(text=f"查到{page}页的相关新闻，总共有{total}条，当前页新闻详情如下，如果有和主题无关的内容，请忽略。"),
                    *[
                        Text(text=f"title={item.title}\nauthor={item.author}\ndescription={item.description}\ncontent={item.fullContent}")
                        for item in news
                    ]
                )]
            ).to_agent_event())


    def as_channel(self):
        chan = PyChannel(name="news", block=True)
        chan.build.command()(self.get_news)
        return chan

class NewsAPIProvider(Provider[NewsAPI]):

    def singleton(self) -> bool:
        return True

    def factory(self, con: IoCContainer) -> INSTANCE:
        eventbus = con.force_fetch(EventBus)
        return NewsAPI(
            eventbus=eventbus,
            api_key=os.environ["NEWSAPI_API_KEY"],
        )


async def main():
    newsapi = NewsAPI(api_key=os.environ["NEWSAPI_API_KEY"])
    news_response = await newsapi.get_news(q="伊朗最新局势")
    print(news_response)

if __name__ == "__main__":
    import os
    # 基于Shadowsocks的代理，翻墙
    # os.environ["HTTP_PROXY"] = "http://127.0.0.1:1087"
    # os.environ["HTTPS_PROXY"] = "http://127.0.0.1:1087"
    os.environ["NEWSAPI_API_KEY"] = "1d8d11c1d95c415a9856c1bc05836076"
    import asyncio
    asyncio.run(main())
