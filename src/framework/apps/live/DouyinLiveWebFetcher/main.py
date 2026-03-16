#!/usr/bin/python
# coding:utf-8
import asyncio
import logging

# @FileName:    main.py
# @Time:        2024/1/2 22:27
# @Author:      bubu
# @Project:     douyinLiveWebFetcher

from framework.apps.live.DouyinLiveWebFetcher.liveMan import DouyinLiveWebFetcher

logging.basicConfig(level=logging.DEBUG)

async def main():
    live_id = '969723624632'
    room = DouyinLiveWebFetcher(live_id, logger=logging.getLogger(__name__))
    # room.get_room_status() # 失效
    await room.start()

if __name__ == '__main__':
    asyncio.run(main())