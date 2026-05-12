from ghoshell_common.contracts import LoggerItf
from ghoshell_container import get_container, Container
from ghoshell_moss import Speech
from ghoshell_moss.speech import BaseTTSSpeech
from ghoshell_moss.speech.player.pyaudio_player import PyAudioStreamPlayer
from framework.speech.xiaomi_tts import XiaomiTTS, XiaomiTTSConf


def get_example_speech(
    container: Container | None = None,
) -> Speech:
    container = container or get_container()
    logger = container.get(LoggerItf)
    return BaseTTSSpeech(
        player=PyAudioStreamPlayer(logger=logger),
        tts=XiaomiTTS(logger=logger),
        logger=logger,
    )