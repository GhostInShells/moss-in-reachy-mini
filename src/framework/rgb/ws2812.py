import asyncio
import logging
import random
import time
from enum import Enum
from typing import Optional

import serial_asyncio

from ghoshell_common.contracts import LoggerItf
from ghoshell_common.helpers import uuid
from ghoshell_container import IoCContainer
from ghoshell_moss import Channel, ChannelRuntime, PyChannel, CommandError

from framework.rgb.serial import auto_detect_port


class BPMFlashMode(Enum):
    """BPM闪烁模式"""
    ALL_FLASH = 1      # 全闪
    ALTERNATE = 2      # 交替
    RUNNING = 3        # 跑动
    GRADIENT = 4       # 渐变


class WS2812Channel(Channel):
    def __init__(
        self,
        name: str,
        description: str,
        baudrate: int = 115200,
        led_count: int = 55,
        serial_timeout: float = 2.0,
        command_delay: float = 0.1,
        logger: Optional[LoggerItf] = None,
    ):
        self._id = uuid()
        self._name = name
        self._description = description

        self._baudrate = baudrate
        self._led_count = led_count
        self._serial_timeout = serial_timeout
        self._command_delay = command_delay

        self._runtime: Optional[ChannelRuntime] = None
        self._serial_writer: Optional[asyncio.StreamWriter] = None
        self._serial_reader: Optional[asyncio.StreamReader] = None

        self.logger = logger or logging.getLogger("WS2812Channel")

    def name(self) -> str:
        return self._name

    def id(self) -> str:
        return self._id

    def description(self) -> str:
        return self._description

    def bootstrap(self, container: Optional[IoCContainer] = None) -> "ChannelRuntime":
        if self._runtime and self._runtime.is_available():
            return self._runtime

        chan = PyChannel(name=self.name(), description=self.description())
        chan.build.command()(self.set_led)
        chan.build.command()(self.set_all)
        chan.build.command()(self.fill_color)
        chan.build.command()(self.clear)

        chan.build.command()(self.rainbow)
        chan.build.command()(self.breath)
        chan.build.command()(self.bpm_flash)

        chan.build.start_up(self.on_start_up)
        chan.build.close(self.on_close)

        self._runtime = chan.bootstrap(container)
        return self._runtime

    async def on_start_up(self):
        port = auto_detect_port()
        if not port:
            self.logger.warning("WS2812: serial port not found, RGB commands will be no-ops")
            return

        try:
            reader, writer = await serial_asyncio.open_serial_connection(
                url=port, baudrate=self._baudrate
            )
            self._serial_writer = writer
            self._serial_reader = reader
        except Exception:
            self.logger.warning("WS2812: failed to open serial port %s, RGB commands will be no-ops", port)

    async def on_close(self):
        await self.clear()

    async def send_command(self, cmd):
        """发送命令到串口"""
        if self._serial_writer is None:
            return
        try:
            self._serial_writer.write((cmd + "\r\n").encode('utf-8'))
            await self._serial_writer.drain()
        except Exception:
            pass

    async def set_led(self, index: int, r: int, g: int, b: int):
        """
        设置单个LED颜色

        :param index: LED灯珠索引
        :param r: RGB的R，0-255
        :param g: RGB的G，0-255
        :param b: RGB的B，0-255
        """
        if index < 0 or index >= self._led_count:
            raise CommandError(
                message=f"错误：LED索引必须在0-{self._led_count-1}之间"
            )
        if not (0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255):
            raise CommandError(
                message="错误：颜色值必须在0-255之间"
            )

        cmd = f"LED {index} {r} {g} {b}"
        return await self.send_command(cmd)

    async def set_all(self, r, g, b):
        """
        设置所有LED颜色

        :param r: RGB的R，0-255
        :param g: RGB的G，0-255
        :param b: RGB的B，0-255
        """
        if not (0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255):
            raise CommandError(
                message="错误：颜色值必须在0-255之间"
            )

        cmd = f"ALL {r} {g} {b}"
        return await self.send_command(cmd)

    async def fill_color(self, start, end, r, g, b):
        """
        设置所有LED颜色

        :param start: LED灯珠的索引
        :param end: LED灯珠的索引
        :param r: RGB的R，0-255
        :param g: RGB的G，0-255
        :param b: RGB的B，0-255
        """
        if start < 0 or start >= self._led_count or end < 0 or end >= self._led_count or start > end:
            raise CommandError(
                message=f"错误：LED范围无效，必须在0-{self._led_count-1}之间且start≤end"
            )
        if not (0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255):
            raise CommandError(
                message="错误：颜色值必须在0-255之间"
            )

        cmd = f"FILL {start} {end} {r} {g} {b}"
        return await self.send_command(cmd)

    async def clear(self):
        """
        清除所有LED
        """
        return await self.send_command("CLEAR")

    async def rainbow(self, speed: float=0.1, duration: float=5.0):
        """
        彩虹渐变
        """
        asyncio.create_task(self._rainbow_task(speed, duration))

    async def _rainbow_task(self, speed: float, duration: float):
        start = time.time()
        try:
            while time.time() - start < duration:
                # 生成彩虹颜色（从红到紫）
                for hue in range(0, 360, 360 // self._led_count):
                    # 将HSV转换为RGB
                    h = hue / 60.0
                    x = int(255 * (1 - abs(h % 2 - 1)))

                    if 0 <= h < 1:
                        r, g, b = 255, x, 0
                    elif 1 <= h < 2:
                        r, g, b = x, 255, 0
                    elif 2 <= h < 3:
                        r, g, b = 0, 255, x
                    elif 3 <= h < 4:
                        r, g, b = 0, x, 255
                    elif 4 <= h < 5:
                        r, g, b = x, 0, 255
                    else:
                        r, g, b = 255, 0, x

                    # 设置所有LED为当前彩虹颜色
                    await self.send_command(f"ALL {r} {g} {b}")
                    await asyncio.sleep(speed)

        except Exception as e:
            print(f"彩虹效果异常: {e}")
        finally:
            await self.clear()

    async def breath(self, r: int, g: int, b: int, breath_time: float=2.0, duration: float=5.0):
        """
        呼吸灯
        """
        asyncio.create_task(self._breath_task(r, g, b, breath_time, duration))

    async def _breath_task(self, r: int, g: int, b: int, breath_time: float, duration: float):
        start = time.time()
        try:
            while time.time() - start < duration:
                # 淡入（0-255）
                steps = int(breath_time / 0.1) / 2
                if steps > 0:
                    for brightness in range(0, 256, max(1, int(256/steps))):

                        r_adj = int(r * brightness / 255)
                        g_adj = int(g * brightness / 255)
                        b_adj = int(b * brightness / 255)
                        await self.send_command(f"ALL {r_adj} {g_adj} {b_adj}")
                        await asyncio.sleep(breath_time / (2 * steps))

                # 淡出（255-0）
                for brightness in range(255, -1, -max(1, int(256/steps))):
                    r_adj = int(r * brightness / 255)
                    g_adj = int(g * brightness / 255)
                    b_adj = int(b * brightness / 255)
                    await self.send_command(f"ALL {r_adj} {g_adj} {b_adj}")
                    await asyncio.sleep(breath_time / (2 * steps))

            await self.clear()

        except Exception as e:
            print(f"呼吸灯效果异常: {e}")
        finally:
            await self.clear()

    async def bpm_flash(self, bpm: int=120, mode: str="", duration: float=5.0):
        """
        :param bpm: bpm
        :param mode: all, alter, running, gradient
        :param duration: 时长
        """
        asyncio.create_task(self._bpm_flash_task(bpm, mode, duration))

    async def _bpm_flash_task(self, bpm: int, mode: str, duration: float):
        beat_interval = 60.0 / bpm
        start = time.time()
        try:
            beat_count = 0
            while time.time() - start < duration:
                beat_count += 1

                if mode == "all":
                    # 每拍生成随机颜色
                    if beat_count % 2 == 1:
                        r1, g1, b1 = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
                        await self.send_command(f"ALL {r1} {g1} {b1}")
                    else:
                        r2, g2, b2 = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
                        await self.send_command(f"ALL {r2} {g2} {b2}")

                elif mode == "alter":
                    # 每拍生成两个随机颜色
                    r1, g1, b1 = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
                    r2, g2, b2 = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)

                    for i in range(self._led_count):
                        if i % 2 == (beat_count % 2):
                            await self.send_command(f"LED {i} {r1} {g1} {b1}")
                        else:
                            await self.send_command(f"LED {i} {r2} {g2} {b2}")

                elif mode == "running":
                    # 跑动模式：节拍点移动
                    pos = (beat_count - 1) % self._led_count
                    r1, g1, b1 = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
                    await self.send_command(f"LED {pos} {r1} {g1} {b1}")

                elif mode == "gradient":
                    # 渐变模式：亮度随节拍变化
                    r1, g1, b1 = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)

                    brightness = 0.5 + 0.5 * (beat_count % 2)  # 交替50%和100%
                    r_adj = int(r1 * brightness)
                    g_adj = int(g1 * brightness)
                    b_adj = int(b1 * brightness)
                    await self.send_command(f"ALL {r_adj} {g_adj} {b_adj}")

                # 等待下一拍
                await asyncio.sleep(beat_interval)

        except Exception as e:
            print(f"BPM效果异常: {e}")
        finally:
            await self.clear()