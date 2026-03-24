import asyncio
import json
from typing import Optional

import aiohttp
from ghoshell_common.helpers import uuid
from ghoshell_container import IoCContainer, Provider, INSTANCE
from ghoshell_moss import Channel, ChannelRuntime, PyChannel, Message, Text
from ghoshell_moss.core.concepts.command import CommandTaskResult


class VolcWebsearchChannel(Channel):
    def __init__(self, name: str, description: str, api_key: str):
        self._name = name
        self._description = description
        self._api_key = api_key
        self._id = uuid()

        self._runtime: Optional[ChannelRuntime] = None

    def name(self) -> str:
        return self._name

    def id(self) -> str:
        return self._id

    def description(self) -> str:
        return self._description

    async def websearch(self, query: str) -> CommandTaskResult:
        """
        异步调用 feedcoopapi 的 web_search 接口

        Args:
            query: 搜索关键词（如"伊朗局势"）

        Returns:
            dict: 接口返回的 JSON 数据
        """
        # 接口URL
        url = "https://open.feedcoopapi.com/search_api/web_search"

        # 请求头配置
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}"
        }

        # 请求体数据
        payload = {
            "Query": query,
            "SearchType": "web",
            "Count": 3,
        }

        # 创建异步HTTP会话并发送请求
        async with aiohttp.ClientSession() as session:
            try:
                # 发送POST请求
                async with session.post(
                        url=url,
                        headers=headers,
                        json=payload  # aiohttp会自动将dict转为JSON字符串，无需手动json.dumps
                ) as response:
                    # 检查响应状态码
                    if response.status != 200:
                        raise Exception(f"请求失败，状态码: {response.status}, 响应内容: {await response.text()}")

                    # 解析JSON响应
                    result = await response.json()
                    res = []
                    for item in result.get("Result", {}).get("WebResults", []):
                        res.append({
                            "title": item["Title"],
                            "content": item["Content"],
                            "publish_time": item["PublishTime"],
                            "auth_info_des": item["AuthInfoDes"],
                        })

                    return CommandTaskResult(
                        result=res,
                        messages=[Message.new(role="user", name="__websearch_result__").with_content(
                            Text(text="基于搜索结果进行**整合分析**，而不仅仅是转述；以及此次结果react行为避免**继续搜索**。"),
                        )],
                        observe=True,
                    )

            except aiohttp.ClientError as e:
                raise Exception(f"网络请求错误: {str(e)}")
            except json.JSONDecodeError as e:
                raise Exception(f"响应JSON解析错误: {str(e)}")
            except Exception as e:
                raise Exception(f"请求异常: {str(e)}")

    def bootstrap(self, container: Optional[IoCContainer] = None) -> "ChannelRuntime":
        if self._runtime is not None and self._runtime.is_running():
            return self._runtime

        chan = PyChannel(name=self.name(), description=self.description(), blocking=False)
        chan.build.command()(self.websearch)

        self._runtime = chan.bootstrap(container=container)
        return self._runtime
