from framework.listener.concepts import *

from framework.listener.callbacks import ConsoleCallback, LoggerCallback, AsyncLoggerCallback, AsyncConsoleCallback

# 常用工具类导出
from framework.listener.chat.console_ptt import ConsolePTTChat

# 异步模块导出
try:
    from framework.listener.async_concepts import (
        AsyncAudioInput,
        AsyncRecognitionCallback,
        AsyncRecognitionBatch,
        AsyncRecognizer,
        AsyncListenerCallback,
        AsyncListenerState,
        AsyncListenerService,
        AsyncListenerStateName,
        Recognition,
    )

    # 异步服务实现
    from framework.listener.async_listener_service import AsyncListenerServiceImpl

    # 异步状态实现
    from framework.listener.async_states import (
        AsyncAudioInputLoop,
        AsyncDeafState,
        AsyncListeningState,
        AsyncPdtListeningState,
        AsyncPdtWaitingState,
    )

    # 异步音频输入
    from framework.listener.async_pyaudio_input import AsyncPyAudioInput, AsyncPyAudioInputConfig

    # 异步ASR
    from framework.listener.async_volcengine_bm import (
        AsyncVocEngineBigModelASR,
        AsyncVocEngineBigModelStreamASRBatch,
    )

    # 注意：ConsolePTTChat 已在 console_ptt.py 中更新为使用异步接口
    # 可以从 framework.listener.chat.console_ptt 导入 ConsolePTTChat

    __all_async__ = [
        'AsyncAudioInput',
        'AsyncRecognitionCallback',
        'AsyncRecognitionBatch',
        'AsyncRecognizer',
        'AsyncListenerCallback',
        'AsyncListenerState',
        'AsyncListenerService',
        'AsyncListenerStateName',
        'Recognition',
        'AsyncListenerServiceImpl',
        'AsyncAudioInputLoop',
        'AsyncDeafState',
        'AsyncListeningState',
        'AsyncPdtListeningState',
        'AsyncPdtWaitingState',
        'AsyncPyAudioInput',
        'AsyncPyAudioInputConfig',
        'AsyncVocEngineBigModelASR',
        'AsyncVocEngineBigModelStreamASRBatch',
        # 注意：ConsolePTTChat 现在是同步类但使用异步内部实现
    ]

except ImportError as e:
    # 异步模块可能不完整，不影响现有功能
    import logging
    logging.getLogger(__name__).debug(f"Async modules not fully available: {e}")
