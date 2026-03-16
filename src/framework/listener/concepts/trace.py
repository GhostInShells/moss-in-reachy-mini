from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, Dict

from ghoshell_common.contracts import LoggerItf
from ghoshell_container import get_caller_info, Container, Provider

_ID = str
_LOG = str

__all__ = ['Tracer', 'LoggerTracer', 'LoggerTracerProvider', 'Tag']


def trace_log(tid: _ID, log: _LOG = "", *args) -> None:
    """
    添加日志到 tracer 的函数.
    :param tid: 贯穿 tracer 的 id.
    :param log: 日志讯息.
    :param args: 日志的参数, 最终会生成 log % args 的日志.
    """
    pass


class TagTracer(ABC):
    """
    某个 tag 下的 tracer.
    """

    @abstractmethod
    def record(self, point: str) -> "trace_log":
        """
        生成一个 trace log 函数, 用来记录固定位置的信息.
        :param point: 用来区分相同 tag 下不同的位置.
        """
        pass


class Tracer(ABC):
    """
    链路排查工具.
    """

    @abstractmethod
    def trace(self, tag: str) -> TagTracer:
        pass


class Tag(str, Enum):
    """
    监控的标签.
    """
    AGENT_EVENT = "AgentEvent"
    SPEECH_BATCH = "SpeechBatch"
    AUDIO_PLAY = "AudioPlay"
    ASR_BATCH = "ASRBatch"


class _Recorder:
    def __init__(self, logger: LoggerItf, tag: str, point: str, call_at: str):
        self._logger = logger
        self._tag = tag
        self._point = point
        self._call_at = call_at
        self._prefix = f"[Trace] - [{self._tag}] - [{self._point}] - [{self._call_at}]"

    def __call__(self, _id: _ID, log: _LOG = "", *args) -> None:
        self._logger.info(f"%s on id %s: {log}", self._prefix, _id, *args)


class LoggerTagTracer(TagTracer):

    def __init__(self, logger: LoggerItf, tag: str):
        self.logger = logger
        self.tag = str(tag)
        self.recorders: Dict[str, "trace_log"] = {}

    def record(self, point: str) -> "trace_log":
        if point in self.recorders:
            return self.recorders[point]

        call_at = get_caller_info(2, with_full_file=False)
        recorder = _Recorder(self.logger, self.tag, point, call_at)
        self.recorders[point] = recorder.__call__
        return recorder


class LoggerTracer(Tracer):

    def __init__(self, logger: LoggerItf):
        self.logger = logger

    def trace(self, tag: str) -> TagTracer:
        return LoggerTagTracer(self.logger, tag)


class LoggerTracerProvider(Provider[Tracer]):

    def singleton(self) -> bool:
        return True

    def factory(self, con: Container) -> Optional[Tracer]:
        logger = con.force_fetch(LoggerItf)
        return LoggerTracer(logger)
