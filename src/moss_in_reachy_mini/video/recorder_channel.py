import time

from ghoshell_moss import Message, PyChannel, Text

from moss_in_reachy_mini.video.recorder_worker import VideoRecorderWorker


class VideoRecorder:
    def __init__(self, worker: VideoRecorderWorker):
        self._worker = worker

    async def start_recording(self, note: str = "") -> str:
        """Start background recording (camera + mic + robot output audio)."""
        return self._worker.start_recording(note=note)

    async def stop_recording(self) -> str:
        """Stop recording and return saved file absolute path."""
        return self._worker.stop_recording()

    async def status(self) -> str:
        """Return a human readable recording status."""
        info = self._worker.status()
        if not info.recording:
            last = self._worker.last_result()
            if last is None:
                return "not recording"
            return (
                f"not recording. last_saved={last.saved_path} duration={last.duration_s}s "
                f"audio_out={last.has_audio_out} audio_in={last.has_audio_in}"
            )
        return f"recording {info.file_name} for {int(time.time() - info.started_at)}s"

    async def context_messages(self):
        info = self._worker.status()
        msg = Message.new(role="system", name="__video_recorder__")
        if info.recording:
            msg.with_content(Text(text=f"Video recording is ON: {info.file_name}"))
        else:
            last = self._worker.last_result()
            if last is None:
                msg.with_content(Text(text="Video recording is OFF"))
            else:
                msg.with_content(
                    Text(
                        text=(
                            f"Video recording is OFF. last_saved={last.saved_path} duration={last.duration_s}s "
                            f"audio_out={last.has_audio_out} audio_in={last.has_audio_in} meta={last.meta_path}"
                        )
                    )
                )
        return [msg]

    def as_channel(self) -> PyChannel:
        chan = PyChannel(
            name="video_recorder",
            description="Background video recorder (camera + mic + robot output audio).",
            block=True,
        )

        chan.build.command()(self.start_recording)
        chan.build.command()(self.stop_recording)
        chan.build.command()(self.status)
        chan.build.with_context_messages(self.context_messages)
        return chan
