import logging
import threading
from typing import List
import os

import requests
import trafilatura
from ghoshell_common.contracts import LoggerItf
from ghoshell_container import Provider, IoCContainer, INSTANCE
from ghoshell_moss import PyChannel
from ghoshell_moss.core.concepts.command import CommandTaskResult
from newsapi import NewsApiClient
from pydantic import BaseModel, Field
from trafilatura import downloads

from framework.abcd.agent_hub import EventBus


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
    def __init__(self, api_key: str, eventbus: EventBus=None, logger: LoggerItf=None):
        # newsapi中国需要挂代理才可以访问
        NEWS_HTTP_PROXY = os.getenv("NEWS_HTTP_PROXY")
        NEWS_HTTPS_PROXY = os.getenv("NEWS_HTTPS_PROXY")

        if NEWS_HTTP_PROXY:
            downloads.PROXY_URL = NEWS_HTTP_PROXY

        self._eventbus = eventbus
        session = requests.Session()
        session.proxies = {}
        if NEWS_HTTP_PROXY:
            session.proxies.update({
                "http": NEWS_HTTP_PROXY,
            })
        if NEWS_HTTPS_PROXY:
            session.proxies.update({
                "https": NEWS_HTTPS_PROXY,
            })
        self._session = session
        self.newsapi = NewsApiClient(api_key=api_key, session=session)

        self.logger = logger or logging.getLogger("NewsAPI")

    async def get_news(self, q: str, page_size: int = 10, page: int = 1):
        """
        获取新闻
        :param q: 新闻关键词
        :param page_size: 每页新闻数量
        :param page: 新闻页码
        :return:
        """

        self.logger.info(f"get_news q={q} page_size={page_size} page={page}")

        res = self.newsapi.get_everything(q=q, page_size=page_size, page=page)

        status = res["status"]
        total = res["totalResults"]
        articles = res["articles"]

        news: List[NewsDetails] = []

        # 定义包装函数，用于在线程中执行目标函数
        def wrapper(d: NewsDetails):
            try:
                response = self._session.get(d.url)
                response.raise_for_status()
                d.fullContent = trafilatura.extract(response.content.decode("utf-8"), output_format="markdown")
                news.append(d)
            except Exception as e:
                news.append(d)

        threads = []
        for item in articles:
            details = NewsDetails.model_validate(item)
            thread = threading.Thread(target=wrapper, args=(details,))
            threads.append(thread)

        for thread in threads:
            thread.daemon = True
            thread.start()
            thread.join(timeout=2)

        # if self._eventbus:
        #     await self._eventbus.put(ReactAgentEvent(
        #         messages=[Message.new(role="system").with_content(
        #             Text(text=f"get_news查到{page}页的相关新闻，总共有{total}条，当前页新闻详情如下，如果有和主题无关的内容，请忽略。"),
        #             *[
        #                 Text(text=f"title={item.title}\nauthor={item.author}\ndescription={item.description}\ncontent={item.fullContent}")
        #                 for item in news
        #             ]
        #         )]
        #     ))
        return CommandTaskResult(
            result=[item.fullContent for item in news],
            observe=True
        )

    def as_channel(self):
        chan = PyChannel(name="news", blocking=True)
        chan.build.command()(self.get_news)
        return chan


class NewsAPIProvider(Provider[NewsAPI]):

    def singleton(self) -> bool:
        return True

    def factory(self, con: IoCContainer) -> INSTANCE:
        eventbus = con.force_fetch(EventBus)
        logger = con.get(LoggerItf)
        return NewsAPI(
            eventbus=eventbus,
            api_key=os.environ.get("NEWSAPI_API_KEY"),
            logger=logger,
        )


async def main():
    newsapi = NewsAPI(api_key=os.environ["NEWSAPI_API_KEY"])
    news_response = await newsapi.get_news(q="查询中国当前油价及近期变化")
    print(news_response)

if __name__ == "__main__":
    # 基于Shadowsocks的代理，翻墙
    os.environ["NEWS_HTTP_PROXY"] = "http://127.0.0.1:1087"
    os.environ["NEWS_HTTPS_PROXY"] = "http://127.0.0.1:1087"
    os.environ["NEWSAPI_API_KEY"] = "1d8d11c1d95c415a9856c1bc05836076"
    import asyncio
    asyncio.run(main())
