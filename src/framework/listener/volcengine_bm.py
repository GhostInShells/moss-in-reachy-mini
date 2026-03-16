import asyncio
import json
import time
import uuid
from collections import deque
from threading import Thread, Event as ThreadEvent
from typing import Optional, Union

import numpy as np
import websockets
from ghoshell_common.contracts import LoggerItf
from ghoshell_common.helpers import uuid, Timeleft
from websockets import State

from framework.listener.concepts import Recognition, RecognitionCallback, RecognitionBatch, Recognizer
from framework.listener.concepts.trace import LoggerTracer, Tag
from framework.listener.callbacks import LoggerCallback
from .volcengine_bm_protocol import *


class VocEngineBigModelStreamASRBatch(RecognitionBatch):
    """
    实现一轮 asr.
    asr 可以每次重启一个 connection, 只要音频先开始 buffer, 重建连接的耗时就不重要.
    实现非常糟, 可以考虑重写.
    """

    def __init__(
            self,
            *,
            batch_id: str,
            config: VolcanoBigModelASRConfig,
            callback: RecognitionCallback,
            logger: LoggerItf,
            vad: Optional[int] = None,
            stop_on_sentence: bool = True,
    ):
        if not batch_id:
            batch_id = uuid()
        self.batch_id = batch_id
        self.config = config.resolve_env()
        self.logger = logger
        self.callback = callback
        self._started = False
        self._committed = False
        """音频如果已经输入结束, 并且拿到了下一个分句, 就退出. """

        self._vad = vad
        self._audio_buffer: deque[np.ndarray] = deque()
        # 不用 asyncio queue, 有神秘的性能问题尚不了解原因. 可能和 loop 有关.
        self._audio_queue: deque[Optional[np.ndarray]] = deque()
        """如果音频 item 为 None 表示要结束音频输出. 但直接未解析的还会发送完."""

        self._close_event = asyncio.Event()
        """结束 asr 流程"""

        self._receiving_done = False
        self._sending_done = False

        self._last_recognition: Optional[Recognition] = None
        self._send_audio_seq = 0
        self._receive_rec_seq = 0
        self._main_loop_thread = Thread(target=self._run_main_loop, daemon=True)
        self._main_loop_is_done = ThreadEvent()
        # trace对象
        self._trace_batch = LoggerTracer(self.logger).trace(Tag.ASR_BATCH.value)
        self._stop_on_sentence = stop_on_sentence

    async def _main_loop(self):
        # connect是async函数，需要await
        self._trace_batch.record("main loop start")(self.batch_id)
        try:
            async with (await connect(self.config, self.batch_id)) as ws:
                uid = self.batch_id
                await send_init_request(ws, self.config, uid, vad=self._vad)
                # 发送音频到服务端的 task
                sending = asyncio.create_task(self._send_audio_request(ws))
                # 从服务端获取数据的 task.
                receiving = asyncio.create_task(self._receive_data(ws))
                closed = asyncio.create_task(self._close_event.wait())
                done, pending = await asyncio.wait([sending, receiving, closed], return_when=asyncio.FIRST_COMPLETED)
                if closed in done:
                    self.logger.info("Cancel main loop on close event")
                    sending.cancel()
                    receiving.cancel()
                await sending
                await receiving
        except asyncio.CancelledError:
            self.logger.info("Main loop is cancelled")
        except websockets.exceptions.ConnectionClosed as e:
            self.logger.info("Connection closed: %s", e)
            return
        except Exception as e:
            self.logger.exception(e)
            raise e
        finally:
            self._close_event.set()
            # 防止异常情况无法发送尾包.
            if self._last_recognition and not self._last_recognition.is_last:
                last_recognition = self._last_recognition.model_copy()
                last_recognition.is_last = True
                self.callback.on_recognition(last_recognition)
            self._trace_batch.record("main loop end")(self.batch_id)

    async def _send_single_audio_request(self, ws: websockets.ClientConnection) -> bool:
        """
        发送一帧音频.
        :return: committed
        """
        try:
            while len(self._audio_queue) == 0:
                await asyncio.sleep(0.05)
                if self._committed:
                    self._audio_queue.append(None)

            audio_data = self._audio_queue.popleft()
            if audio_data is None:
                return True
            self._send_audio_seq += 1
            # 如果 commited 设置了, 就发送尾包.
            if self._send_audio_seq == 1:
                self._trace_batch.record(f"send first audio chunk")(self.batch_id)
            self.logger.debug("get new audio audio data at seq: %d", self._send_audio_seq)
            audio_bytes = nparray_to_bytes(audio_data)
            sending_done = self._committed
            await send_audio(ws, audio_bytes, self._send_audio_seq, sending_done)
            self.logger.debug(
                "Sent audio only request at seq: %d", self._send_audio_seq
            )
            return sending_done
        except Exception as e:
            self.logger.exception(e)
        except websockets.exceptions.ConnectionClosed as e:
            self.logger.info("Connection closed: %s", e)
        finally:
            self.logger.debug("sending single data done")

    async def _send_audio_request(self, ws: websockets.ClientConnection) -> bool:
        try:
            while not self._close_event.is_set():
                await self._send_single_audio_request(ws)
        except Exception as e:
            self.logger.exception(e)
            raise e
        finally:
            # 标记数据发送已经结束.
            self._sending_done = True
            self._close_event.set()
            self._trace_batch.record(f"sending audios done")(self.batch_id)

    async def _receive_data(self, ws: websockets.ClientConnection):
        try:
            while not self._close_event.is_set():
                rec = await self._receive_single_resp(ws)
                self.logger.debug("receiving single data done")
                if rec:
                    self._receive_rec_seq += 1
                    # 拿到了一个确定的分句.
                    if rec.sentence:
                        # 判断是否是尾包. 暂时不做序列号校验.
                        # 按豆包模型的逻辑, 上传一个语音包, 下发一个解析包, 所以顺序默认就是对齐的.
                        is_last_audio_rec = self._sending_done and self._receive_rec_seq == self._send_audio_seq
                        rec.is_last = self._stop_on_sentence or is_last_audio_rec
                    self._last_recognition = rec
                    self.callback.on_recognition(rec)
                    if rec.is_last:
                        # 中断循环.
                        self._trace_batch.record("receive last recognition")(self.batch_id)
                        return

        except websockets.exceptions.ConnectionClosed as e:
            self.logger.info("Connection closed: %s", e)
        except Exception as e:
            self.logger.exception(e)
            raise e
        finally:
            self._save_batch_audio()
            self._receiving_done = True
            self._trace_batch.record(f"receiving asr done")(self.batch_id)
            self._close_event.set()

    async def _receive_single_resp(self, ws: websockets.ClientConnection) -> Optional[Recognition]:
        """
        :return: 是否是这个 batch 的最后一个包.
        """
        if ws.state == State.CLOSED:
            return None
        data = await ws.recv()
        if not data:
            self.logger.error("Receiving data failed")
            return None
        self.logger.debug("receive data from server at %s", self.batch_id)
        if isinstance(data, str):
            raise RuntimeError(f'Receiving invalid str data {data}')

        response = parse_response(data)
        self.logger.debug("parsed response from server %s", response)
        if response.message_type == ResponseMessageType.server_error:
            error = "receive server error: %d, %s" % (response.error_code, response.payload)
            self.callback.on_error(error)
            self.logger.error(error)
            # 退出本轮 asr.
            self._close_event.set()
            return None
        elif response.message_type == ResponseMessageType.server_ack:
            self.logger.debug("receive server ack: %s", response.payload)
            return None
        elif response.message_type == ResponseMessageType.full_server_response:
            self.logger.debug("receive server full response: %s", response)
            rec = self._handle_server_full_response(response)
            return rec
        else:
            self.logger.info("receive unknown message: %s", response)
            return None

    def _handle_server_full_response(self, response: Response) -> Optional[Recognition]:
        data = json.loads(response.payload)
        fsp = FullServerResponse(**data)
        if len(fsp.result.utterances) > 0:
            is_sentence = fsp.result.utterances[0].definite
            rec = Recognition(
                batch_id=self.batch_id,
                text=fsp.result.text,
                seq=response.sequence,
                sentence=is_sentence,
                # 尾包是 committed 后拿到的分句.
                is_last=is_sentence and self._committed,
            )
            return rec
        return None

    def _save_batch_audio(self) -> None:
        # todo: 还需要裁剪音频长度.
        if self._last_recognition:
            self.callback.save_batch(self._last_recognition, self.get_buffer())

    def _run_main_loop(self):
        try:
            asyncio.run(self._main_loop())
        finally:
            self._main_loop_is_done.set()

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._main_loop_thread.start()

    def close(self, error: Optional[Exception] = None) -> None:
        self._close_event.set()
        if error is not None:
            self.logger.exception(error)
        self._close_event.set()
        self._trace_batch.record("batch close")(self.batch_id)

    def buffer(self, audio: np.ndarray) -> None:
        """
        提交音频输入到队列里.
        """
        if self._close_event.is_set():
            self.logger.warning("Buffer closed")
            return
        if audio is not None:
            self._audio_buffer.append(audio)
        self._audio_queue.append(audio)

    def commit(self) -> None:
        if self._committed:
            return
        self._committed = True
        self._trace_batch.record("commit batch ready for final")(self.batch_id)

    def get_last_recognition(self) -> Union[Recognition, None]:
        return self._last_recognition

    def get_buffer(self) -> np.ndarray:
        combined_audio = np.concatenate(self._audio_buffer)
        return combined_audio

    def is_done(self) -> bool:
        return self._main_loop_is_done.is_set()

    def wait_until_done(self, timeout: Optional[float] = None) -> None:
        if timeout is None:
            left = Timeleft(0)
        else:
            left = Timeleft(timeout)

        while not self._close_event.is_set() and left.alive():
            if self._main_loop_is_done.wait():
                return
            time.sleep(0.05)


class VocEngineBigModelASR(Recognizer):

    def __init__(
            self,
            *,
            config: VolcanoBigModelASRConfig,
            logger: LoggerItf,
            callback: RecognitionCallback = None,
    ):
        self.config = config.resolve_env()
        self.logger = logger
        self.sample_rate = config.sample_rate
        self.frame_duration = config.frame_time / 1000
        self._closed = False
        self.callback = callback or LoggerCallback(self.logger)

    def start(self) -> None:
        return

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True

    def is_closed(self) -> bool:
        return self._closed

    def new_batch(
            self,
            callback: RecognitionCallback = None,
            batch_id: str = "",
            vad: Optional[int] = None,
            stop_on_sentence: bool = False,
    ) -> RecognitionBatch:
        if callback is None:
            callback = LoggerCallback(self.logger)
        return VocEngineBigModelStreamASRBatch(
            callback=callback, batch_id=batch_id, config=self.config, logger=self.logger,
            vad=vad,
            stop_on_sentence=stop_on_sentence,
        )
