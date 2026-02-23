import asyncio

from ghoshell_moss_contrib.agent import SimpleAgent

from moss_in_reachy_mini import MossInReachyMini


class ReachyMiniAgent(SimpleAgent):
    def __init__(
            self,
            moss_in_reachy_mini: MossInReachyMini,
            instruction: str,
            *args,
            **kwargs
    ):
        self.moss_in_reachy_mini = moss_in_reachy_mini
        super().__init__(instruction, *args, **kwargs)


    async def _response_loop(self, inputs: list[dict]) -> None:
        try:
            if not inputs:
                return
            # 取消当前idle move
            self.logger.info(f"Cancelling idle move in state {self.moss_in_reachy_mini.state.NAME} ")
            await self.moss_in_reachy_mini.state.cancel_idle_move()

            while inputs is not None and not self._interrupt_requested:
                inputs = await self._single_response(inputs)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.exception("Response loop failed")
            self.chat.print_exception(e)
        finally:
            # 开始当前idle move
            self.logger.info(f"Starting idle move in state {self.moss_in_reachy_mini.state.NAME} ")
            await self.moss_in_reachy_mini.state.start_idle_move()

    async def run(self):
        async with self:
            self.moss_in_reachy_mini.set_proactive_input(self.handle_user_input)
            self.chat.set_input_callback(self.handle_user_input)
            self.chat.set_interrupt_callback(self.interrupt)
            await self.chat.run()
        await self.wait_done()