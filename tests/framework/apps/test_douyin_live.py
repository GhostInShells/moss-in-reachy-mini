import pytest
from ghoshell_common.contracts.storage import MemoryStorage

from framework.agent.eventbus import QueueEventBus
from framework.apps.live.douyin_live import DouyinLive, DouyinLiveConfig, DouyinLiveUserHistory, DouyinLiveEvent, \
    DouyinLiveEventType


@pytest.mark.asyncio
async def test_douyin_live():
    live = DouyinLive(
        eventbus=QueueEventBus(),
        history_storage=MemoryStorage(dir_=""),
        config=DouyinLiveConfig(),
    )

    user_id = "mock_id"
    user_name = "mock_name"

    history = await live.get_user_history(
        user_id=user_id,
        user_name=user_name
    )
    assert len(history.history) == 0

    await live.save_user_history(DouyinLiveUserHistory(
        user_id=user_id,
        user_name=user_name,
        history=[
            DouyinLiveEvent(
                user_id=user_id,
                user_name=user_name,
                event_type=DouyinLiveEventType.chat,
                content="hello"
            )
        ]
    ))

    history = await live.get_user_history(
        user_id=user_id,
        user_name=user_name
    )
    assert len(history.history) == 1
    assert len(history.get_history_events()) == 1
    assert len(history.get_history_events(DouyinLiveEventType.chat)) == 1

    history.history.append(
        DouyinLiveEvent(
            user_id=user_id,
            user_name=user_name,
            event_type=DouyinLiveEventType.small_gift,
            content="word"
        )
    )
    await live.save_user_history(history)
    history = await live.get_user_history(
        user_id=user_id,
        user_name=user_name
    )
    assert len(history.history) == 2
    assert len(history.get_history_events()) == 2
    assert len(history.get_history_events(DouyinLiveEventType.chat)) == 1
    assert len(history.get_history_events(DouyinLiveEventType.small_gift)) == 1

