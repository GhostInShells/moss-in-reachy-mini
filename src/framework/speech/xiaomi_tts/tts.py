import asyncio
import logging
import os
from typing import Any, AsyncIterator, ClassVar, Literal, Optional

import numpy as np
from ghoshell_common.contracts import LoggerItf
from ghoshell_common.helpers import uuid
from pydantic import BaseModel, Field

from ghoshell_moss.core.concepts.speech import TTS, AudioFormat, TTSAudioCallback, TTSBatch, TTSInfo, TTSItem
from ghoshell_moss.core.helpers.asyncio_utils import ThreadSafeEvent
from framework.speech.xiaomi_tts.protocol import (
    extract_audio_chunk,
    parse_sse_line,
    pcm16_bytes_to_numpy,
)

__all__ = [
    "SpeakerConf",
    "SpeakerInfo",
    "SpeakerTypes",
    "VoiceConf",
    "XiaomiTTS",
    "XiaomiTTSBatch",
    "XiaomiTTSConf",
]


class SpeakerInfo(BaseModel):
    display_name: str
    language: str
    gender: str

    def description(self) -> str:
        return f"language: {self.language}, gender: {self.gender}"


# Xiaomi MiMo-V2.5-TTS preset voices
SpeakerTypes = Literal[
    "mimo_default",
    "冰糖",
    "茉莉",
    "苏打",
    "白桦",
    "Mia",
    "Chloe",
    "Milo",
    "Dean",
]

SPEAKER_INFO_MAP: dict[SpeakerTypes, SpeakerInfo] = {
    "mimo_default": SpeakerInfo(display_name="MiMo-默认", language="中文", gender="女性"),
    "冰糖": SpeakerInfo(display_name="冰糖", language="中文", gender="女性"),
    "茉莉": SpeakerInfo(display_name="茉莉", language="中文", gender="女性"),
    "苏打": SpeakerInfo(display_name="苏打", language="中文", gender="男性"),
    "白桦": SpeakerInfo(display_name="白桦", language="中文", gender="男性"),
    "Mia": SpeakerInfo(display_name="Mia", language="英文", gender="女性"),
    "Chloe": SpeakerInfo(display_name="Chloe", language="英文", gender="女性"),
    "Milo": SpeakerInfo(display_name="Milo", language="英文", gender="男性"),
    "Dean": SpeakerInfo(display_name="Dean", language="英文", gender="男性"),
}


class VoiceConf(BaseModel):
    style_prompt: Optional[str] = Field(
        default=None,
        description=(
            "Natural language style control prompt. Goes into role:user content. "
            "Examples: '用轻快上扬的语调说话，语速稍快' or 'Warm, friendly tone with a slight laugh'."
        ),
    )


class SpeakerConf(BaseModel):
    """Speaker configuration for Xiaomi TTS."""

    voice_id: str = Field(default="冰糖", description="Voice ID to send to the API")
    description: str = Field(default="", description="Speaker description")
    language: str = Field(default="中文", description="Language")
    gender: str = Field(default="女性", description="Gender")
    voice: VoiceConf = Field(default_factory=VoiceConf, description="Voice style config")


class XiaomiTTSConf(BaseModel):
    """Xiaomi MiMo-V2.5 TTS configuration."""

    api_key: str = Field(
        default="$MIMO_API_KEY",
        description="API key. Prefix with $ to read from env var.",
    )
    base_url: str = Field(
        default="https://api.xiaomimimo.com/v1",
        description="API base URL",
    )
    model: str = Field(
        default="mimo-v2.5-tts",
        description="Model ID. Options: mimo-v2.5-tts, mimo-v2.5-tts-voicedesign, mimo-v2.5-tts-voiceclone",
    )
    sample_rate: int = Field(default=24000, description="Audio sample rate (Xiaomi default is 24kHz)")
    audio_format: Literal["pcm16"] = Field(default="pcm16", description="Audio format for streaming")

    disconnect_on_idle: int = Field(
        default=300,
        description="Seconds to wait before closing idle connection (for parity with Volcengine)",
    )
    timeout: int = Field(default=30, description="HTTP request timeout in seconds")

    speakers: dict[str, SpeakerConf] = Field(
        default_factory=lambda: {
            info.display_name: SpeakerConf(
                voice_id=name,
                description=info.description(),
                language=info.language,
                gender=info.gender,
            )
            for name, info in SPEAKER_INFO_MAP.items()
        },
        description="Available speakers",
    )
    default_speaker: str = Field(
        default="冰糖",
        description="Default speaker name",
    )

    @classmethod
    def unwrap_env(cls, value: str, default: str = "") -> str:
        if value.startswith("$"):
            return os.environ.get(value[1:], default)
        return value or default

    def resolved_api_key(self) -> str:
        return self.unwrap_env(self.api_key)

    def default_speaker_conf(self) -> SpeakerConf:
        conf = self.speakers.get(self.default_speaker)
        if conf is not None:
            return conf.model_copy(deep=True)
        return SpeakerConf()

    def to_tts_info(self, current_tone: str = "") -> TTSInfo:
        return TTSInfo(
            sample_rate=self.sample_rate,
            channels=1,
            audio_format=AudioFormat.PCM_S16LE.value,
            voice_schema=VoiceConf.model_json_schema(),
            tones={key: value.description for key, value in self.speakers.items()},
            current_tone=current_tone or self.default_speaker,
        )


class XiaomiTTSBatch(TTSBatch):
    instance_count: ClassVar[int] = 0

    def __init__(
        self,
        *,
        loop: asyncio.AbstractEventLoop,
        speaker: SpeakerConf,
        batch_id: str = "",
        channels: int,
        audio_format: str,
        sample_rate: int,
        voice: dict | None,
        tone: str,
        logger: LoggerItf,
        callback: Optional[TTSAudioCallback] = None,
    ):
        self.default_speaker = speaker
        self.callback = callback
        self.tone = tone
        self.voice: dict | None = voice
        self.channel = channels
        self.audio_format = audio_format
        self.sample_rate = sample_rate
        self.committed = False
        self._committed_event = ThreadSafeEvent()
        self.done = ThreadSafeEvent()
        self.text_buffer = ""
        self.exception: Optional[Exception] = None
        self._started = ThreadSafeEvent()
        self._running_loop = loop
        self._has_valid_text = False
        self._batch_id = batch_id or uuid()
        self._chunks: asyncio.Queue[np.ndarray | None] = asyncio.Queue()
        self._log_prefix = f"[XiaomiTTSBatch][id={batch_id} voice={self.voice} tone={self.tone}]"
        self._logger = logger
        XiaomiTTSBatch.instance_count += 1

    def speaker(self) -> SpeakerConf:
        conf = self.default_speaker.model_copy()
        if self.voice is not None:
            voice_conf = VoiceConf(**self.voice)
            conf.voice = voice_conf
        return conf

    def __del__(self):
        XiaomiTTSBatch.instance_count -= 1

    async def append(self, audio: np.ndarray) -> None:
        await self._chunks.put(audio)

    def batch_id(self) -> str:
        return self._batch_id

    async def start(self) -> None:
        self._started.set()

    def is_started(self) -> bool:
        return self._started.is_set()

    async def wait_started(self) -> None:
        if self._started.is_set():
            return
        elif self.done.is_set():
            return
        wait_started_task = asyncio.create_task(self._started.wait())
        wait_done_task = asyncio.create_task(self.done.wait())
        done, pending = await asyncio.wait(
            [wait_started_task, wait_done_task],
            return_when=asyncio.FIRST_COMPLETED,
        )
        for t in pending:
            t.cancel()
        await asyncio.gather(wait_started_task, wait_done_task, return_exceptions=True)

    async def items(self) -> AsyncIterator[TTSItem]:
        if not self._started:
            return
        while True:
            audio = await self._chunks.get()
            if audio is None:
                break
            yield TTSItem(
                tone=self.tone,
                voice=self.voice,
                audio_format=self.audio_format,
                channels=self.channel,
                sample_rate=self.sample_rate,
                audio=audio,
                text="",
            )

    def with_callback(self, callback: TTSAudioCallback) -> None:
        self.callback = callback

    def fail(self, reason: str) -> None:
        self.exception = RuntimeError(reason)
        self.done.set()
        self.commit()

    def feed(self, text: str):
        if self.done.is_set():
            return
        self.text_buffer += text
        if self._has_valid_text:
            self._logger.debug("%s feed text `%s`", self._log_prefix, text)
        elif stripped := self.text_buffer.lstrip():
            self._logger.debug("%s feed first legal text `%s`", self._log_prefix, stripped)
            self._has_valid_text = True

    def commit(self):
        self.committed = True
        self._committed_event.set()
        self._logger.info("%s batch committed", self._log_prefix)

    def is_closed(self) -> bool:
        return self.done.is_set()

    def is_committed(self) -> bool:
        return self.committed

    async def wait_committed(self) -> None:
        if self.committed:
            return
        if self.done.is_set():
            return
        wait_committed_task = asyncio.create_task(self._committed_event.wait())
        wait_done_task = asyncio.create_task(self.done.wait())
        done, pending = await asyncio.wait(
            [wait_committed_task, wait_done_task],
            return_when=asyncio.FIRST_COMPLETED,
        )
        for t in pending:
            t.cancel()
        await asyncio.gather(wait_committed_task, wait_done_task, return_exceptions=True)

    async def close(self) -> None:
        if self.done.is_set():
            return
        self.commit()
        self.done.set()
        self._logger.info("%s batch close. instances count: %d", self._log_prefix, self.instance_count)
        self._chunks.put_nowait(None)

    async def wait_done(self, timeout: float | None = None):
        if timeout is not None and timeout > 0.0:
            await asyncio.wait_for(self.done.wait(), timeout=timeout)
        else:
            await self.done.wait()
        if self.exception is not None:
            raise self.exception


class XiaomiTTS(TTS):
    """Xiaomi MiMo-V2.5 TTS implementation using OpenAI-compatible HTTP API with SSE streaming."""

    def __init__(
        self,
        *,
        conf: XiaomiTTSConf | None = None,
        logger: LoggerItf | None = None,
    ):
        self.logger = logger or logging.getLogger("moss")
        self._log_prefix = "[XiaomiTTS] "

        # --- config state --- #
        self._conf = conf or XiaomiTTSConf()
        self._current_speaker: str = self._conf.default_speaker
        self._current_speaker_conf: SpeakerConf = self._conf.default_speaker_conf()

        # --- runtime --- #
        self._starting = False
        self._started = False
        self._running_loop: Optional[asyncio.AbstractEventLoop] = None
        self._main_loop_task: Optional[asyncio.Task] = None
        self._closing_event = ThreadSafeEvent()
        self._closed_event = ThreadSafeEvent()

        self._pending_batches_queue: asyncio.Queue[XiaomiTTSBatch | None] = asyncio.Queue()
        self._has_any_batch_event = asyncio.Event()

        self._default_tts_info = self.get_info()

    def get_info(self) -> TTSInfo:
        return self._conf.to_tts_info(self._current_speaker)

    def use_tone(self, config_key: str) -> None:
        if config_key not in self._conf.speakers:
            raise LookupError(f"The voice {config_key} not found")
        conf = self._conf.speakers[config_key]
        self.logger.info("%s Using tone %s", self._log_prefix, config_key)
        self._current_speaker = config_key
        self._current_speaker_conf = conf.model_copy(deep=True)

    def current_tone(self) -> str:
        return self._current_speaker

    def set_voice(self, config: dict[str, Any]) -> None:
        voice = VoiceConf(**config)
        self._current_speaker_conf.voice = voice
        self.logger.info("%s set current voice %s", self._log_prefix, config)

    def get_voice(self) -> dict[str, Any]:
        return self._current_speaker_conf.voice.model_dump()

    def _check_running(self) -> None:
        if not self._started or self._closing_event.is_set():
            raise RuntimeError("TTS is closed")

    def new_batch(
        self,
        batch_id: str = "",
        *,
        callback: TTSAudioCallback | None = None,
        voice: dict[str, Any] | None = None,
        tone: str | None = None,
    ) -> TTSBatch:
        self._check_running()
        self.logger.info("%s create new tts batch %s", self._log_prefix, batch_id)
        batch = self._create_batch(batch_id, callback, voice, tone)
        self._pending_batches_queue.put_nowait(batch)
        self._has_any_batch_event.set()
        return batch

    def _create_batch(
        self,
        batch_id: str = "",
        callback: TTSAudioCallback | None = None,
        voice: dict[str, Any] | None = None,
        tone: str | None = None,
    ) -> XiaomiTTSBatch:
        speaker_conf = self._current_speaker_conf
        if tone is not None and tone != self.current_tone():
            speaker_conf = self._conf.speakers.get(tone, speaker_conf)
        return XiaomiTTSBatch(
            loop=self._running_loop,
            speaker=speaker_conf,
            voice=voice,
            tone=tone or speaker_conf.voice_id,
            batch_id=batch_id,
            callback=callback,
            logger=self.logger,
            audio_format=self._default_tts_info.audio_format,
            channels=self._default_tts_info.channels,
            sample_rate=self._default_tts_info.sample_rate,
        )

    async def _main_loop(self):
        """Main loop consuming pending batches."""
        while not self._closing_event.is_set():
            batch = None
            try:
                await self._has_any_batch_event.wait()
                if self._pending_batches_queue.empty():
                    self._has_any_batch_event.clear()
                    continue
                batch = await self._pending_batches_queue.get()
                if batch is None:
                    break
                await self._consume_batch(batch)
            except asyncio.CancelledError:
                self.logger.info("%s TTS cancelled", self._log_prefix)
                break
            except Exception as e:
                self.logger.error("%s TTS main loop got exception: %s", self._log_prefix, e)
            finally:
                if batch is not None and not batch.is_closed():
                    await batch.close()

    async def _consume_batch(self, batch: XiaomiTTSBatch):
        """Synthesize a single batch via HTTP SSE."""
        batch_id = batch.batch_id()
        try:
            await batch.wait_started()
            if batch.is_closed():
                return

            # Wait until all text is fed (commit called) before reading.
            await batch.wait_committed()
            if batch.is_closed():
                return

            speaker = batch.speaker()
            text = batch.text_buffer.strip()
            if not text:
                self.logger.warning("%s batch %s has no text", self._log_prefix, batch_id)
                return

            self.logger.info("%s synthesizing batch %s, text length=%d", self._log_prefix, batch_id, len(text))
            await self._synthesize(batch, speaker, text)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self.logger.exception("%s batch %s failed: %s", self._log_prefix, batch_id, e)
            batch.fail(str(e))
        finally:
            if not batch.is_closed():
                await batch.close()

    async def _synthesize(self, batch: XiaomiTTSBatch, speaker: SpeakerConf, text: str):
        """Make HTTP request and parse SSE response."""
        import aiohttp

        api_key = self._conf.resolved_api_key()
        if not api_key:
            batch.fail("MIMO_API_KEY is not set")
            return

        # Build messages
        messages = []
        # Style instruction in user message
        style_prompt = speaker.voice.style_prompt if speaker.voice else None
        if style_prompt:
            messages.append({"role": "user", "content": style_prompt})
        # Text to synthesize in assistant message
        messages.append({"role": "assistant", "content": text})

        # Build request body
        body = {
            "model": self._conf.model,
            "messages": messages,
            "audio": {
                "format": self._conf.audio_format,
                "voice": speaker.voice_id,
            },
            "stream": True,
        }

        headers = {
            "api-key": api_key,
            "Content-Type": "application/json",
        }

        url = f"{self._conf.base_url}/chat/completions"
        batch_id = batch.batch_id()
        callback = batch.callback

        try:
            timeout = aiohttp.ClientTimeout(total=self._conf.timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=body, headers=headers) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        batch.fail(f"HTTP {resp.status}: {error_text[:500]}")
                        return

                    first = True
                    buffer = ""
                    async for chunk in resp.content.iter_any():
                        if batch.is_closed():
                            break

                        buffer += chunk.decode("utf-8", errors="replace")
                        while "\n" in buffer:
                            line, buffer = buffer.split("\n", 1)
                            event = parse_sse_line(line)
                            if event is None:
                                continue
                            if event.is_done:
                                self.logger.info("%s batch %s SSE done", self._log_prefix, batch_id)
                                break
                            if event.is_empty:
                                continue

                            event_json = event.parse_json()
                            if event_json is None:
                                continue

                            audio_bytes = extract_audio_chunk(event_json)
                            if audio_bytes and len(audio_bytes) > 0:
                                if first:
                                    self.logger.info(
                                        "%s batch %s received first audio chunk, size=%d",
                                        self._log_prefix, batch_id, len(audio_bytes),
                                    )
                                    first = False
                                np_data = pcm16_bytes_to_numpy(audio_bytes)
                                if callback:
                                    callback(np_data)
                                await batch.append(np_data)

        except aiohttp.ClientError as e:
            batch.fail(f"HTTP client error: {e}")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self.logger.exception("%s synthesize failed: %s", self._log_prefix, e)
            batch.fail(str(e))

    async def clear(self) -> None:
        self._check_running()
        self._has_any_batch_event.clear()
        old_queue = self._pending_batches_queue
        self._pending_batches_queue = asyncio.Queue()
        while not old_queue.empty():
            batch = await old_queue.get()
            if batch is not None:
                await batch.close()

    async def start(self) -> None:
        if self._starting:
            return
        self._starting = True
        self._running_loop = asyncio.get_running_loop()
        self._main_loop_task = asyncio.create_task(self._main_loop())
        self._started = True

    async def close(self) -> None:
        if self._closing_event.is_set():
            return
        self.logger.info("%s closing...", self._log_prefix)
        self._closing_event.set()
        self._pending_batches_queue.put_nowait(None)
        if self._main_loop_task is not None:
            self._main_loop_task.cancel()
            try:
                await self._main_loop_task
            except asyncio.CancelledError:
                pass
            finally:
                self._main_loop_task = None
        self._closed_event.set()
