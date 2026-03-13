#!/usr/bin/python
# coding:utf-8
import asyncio

# @FileName:    main.py
# @Time:        2024/1/2 22:27
# @Author:      bubu
# @Project:     douyinLiveWebFetcher

from framework.apps.live.DouyinLiveWebFetcher.liveMan import DouyinLiveWebFetcher


async def main():
    live_id = '972559747665'
    room = DouyinLiveWebFetcher(live_id)
    # room.get_room_status() # 失效
    await room.start()

if __name__ == '__main__':
    asyncio.run(main())