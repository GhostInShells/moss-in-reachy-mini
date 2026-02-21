from typing import Optional, Iterable, Type

from ghoshell_common.contracts import Configs
from ghoshell_common.contracts import LoggerItf
from ghoshell_container import Container, Provider, BootstrapProvider, ABSTRACT, INSTANCE
from reachy_mini import ReachyMini

from listener.concepts.listener import (
    AudioInput, Recognizer, ListenerCallback, RecognitionCallback, ListenerService
)
from listener.callbacks import LoggerCallback
from listener.configs import ListenerConfig
from listener.lisenter_impl import ListenerServiceImpl

__all__ = [
    'ListenerAudioInputProvider', 'ListenerRecognizerProvider', 'LoggerListenerCallbackProvider',
    'ListenerServiceProvider',
]

from listener.mini_input_impl import ReachyMiniInput


class ListenerAudioInputProvider(Provider[AudioInput]):
    """
    音频输入的封装.
    """

    def __init__(self, mini: ReachyMini):
        self.mini = mini


    def singleton(self) -> bool:
        return False

    def factory(self, con: Container) -> Optional[AudioInput]:
        return ReachyMiniInput(self.mini)
        configs = con.force_fetch(Configs)
        conf = configs.get_or_create(ListenerConfig())
        conf = conf.resolve_env()
        audio_input_config = conf.get_audio_input_config()
        logger = con.get(LoggerItf)
        return audio_input_config.new_audio_input(logger=logger)


class LoggerListenerCallbackProvider(Provider[ListenerCallback]):
    """
    日志 provider.
    """

    def singleton(self) -> bool:
        return False

    def aliases(self) -> Iterable[ABSTRACT]:
        yield LoggerCallback
        yield RecognitionCallback

    def factory(self, con: Container) -> Optional[LoggerCallback]:
        logger = con.get(LoggerItf)
        return LoggerCallback(logger=logger)


class ListenerRecognizerProvider(Provider[Recognizer]):
    """
    基于本地配置的 asr
    """

    def __init__(self, asr_name: str = ""):
        self.asr_name = asr_name

    def singleton(self) -> bool:
        return True

    def factory(self, con: Container) -> Optional[Recognizer]:
        configs = con.force_fetch(Configs)
        conf = configs.get_or_create(ListenerConfig())
        conf = conf.model_copy(deep=True)
        if self.asr_name:
            conf.use_asr = self.asr_name

        logger = con.get(LoggerItf)
        callback = con.get(RecognitionCallback)
        reg = ListenerServiceImpl.make_recognizer(conf, logger, callback)
        return reg


class ListenerServiceProvider(BootstrapProvider[ListenerService]):

    def __init__(self, default_state_name: str = ""):
        self.default_state_name = default_state_name

    def singleton(self) -> bool:
        return True

    def contract(self) -> Type[INSTANCE]:
        return ListenerService

    def bootstrap(self, container: Container) -> None:
        service = container.force_fetch(ListenerServiceImpl)
        service.bootstrap()
        container.add_shutdown(service.shutdown)

    def aliases(self) -> Iterable[ABSTRACT]:
        yield ListenerServiceImpl

    def factory(self, con: Container) -> Optional[ListenerService]:
        configs = con.force_fetch(Configs)
        conf = configs.get_or_create(ListenerConfig())
        conf = conf.model_copy(deep=True)
        logger = con.force_fetch(LoggerItf)
        callback = con.get(ListenerCallback)
        return ListenerServiceImpl(
            config=conf,
            logger=logger,
            callback=callback,
            default_state_name=self.default_state_name,
        )
