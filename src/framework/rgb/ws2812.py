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
        self._current_task: Optional[asyncio.Task] = None

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
        await self._stop_current_task()
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
        await self.send_command(f"LED {index} {r} {g} {b}")

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
        await self.send_command(f"ALL {r} {g} {b}")

    async def fill_color(self, start, end, r, g, b):
        """
        设置范围LED颜色

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
        await self.send_command(f"FILL {start} {end} {r} {g} {b}")

    async def clear(self):
        """
        清除所有LED
        """
        await self.send_command("CLEAR")

    async def _stop_current_task(self) -> None:
        if self._current_task and not self._current_task.done():
            self._current_task.cancel()
            try:
                await self._current_task
            except (asyncio.CancelledError, Exception):
                pass
        self._current_task = None

    async def rainbow(self, speed: float = 0.1, duration: float = 5.0):
        """
        彩虹渐变
        """
        await self._stop_current_task()
        self._current_task = asyncio.create_task(self._rainbow_task(speed, duration))

    async def _rainbow_task(self, speed: float, duration: float):
        start = time.time()
        try:
            while time.time() - start < duration:
                for hue in range(0, 360, 360 // self._led_count):
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
                    await self.send_command(f"ALL {r} {g} {b}")
                    await asyncio.sleep(speed)
        except Exception as e:
            print(f"彩虹效果异常: {e}")
        finally:
            await self.clear()

    async def breath(self, r: int, g: int, b: int, breath_time: float = 2.0, duration: float = 5.0):
        """
        呼吸灯
        """
        await self._stop_current_task()
        self._current_task = asyncio.create_task(self._breath_task(r, g, b, breath_time, duration))

    async def _breath_task(self, r: int, g: int, b: int, breath_time: float, duration: float):
        start = time.time()
        try:
            while time.time() - start < duration:
                steps = int(breath_time / 0.1) / 2
                if steps > 0:
                    for brightness in range(0, 256, max(1, int(256 / steps))):
                        await self.send_command(f"ALL {int(r*brightness/255)} {int(g*brightness/255)} {int(b*brightness/255)}")
                        await asyncio.sleep(breath_time / (2 * steps))
                for brightness in range(255, -1, -max(1, int(256 / steps))):
                    await self.send_command(f"ALL {int(r*brightness/255)} {int(g*brightness/255)} {int(b*brightness/255)}")
                    await asyncio.sleep(breath_time / (2 * steps))
        except Exception as e:
            print(f"呼吸灯效果异常: {e}")
        finally:
            await self.clear()

    async def bpm_flash(self, bpm: int = 120, mode: str = "", duration: float = 5.0):
        """        BPM同步灯光效果。

        :param bpm: 节拍速度
        :param mode: 效果模式：
            - all: 全体随机颜色闪烁，每拍换色，最强烈
            - alter: 全体同色，每拍换随机色，配合节奏律动
            - running: 单点跑马灯，每拍点亮下一颗，逐渐铺满整条灯带
            - gradient: 全体同色亮度50%/100%交替，柔和律动
            - meteor: 4路流星，每拍各移动3格，逐步铺满
            - sparkle: 每拍随机点亮3颗，散点积累闪烁
            - bounce: 2路彩球从两端向中间移动，逐步铺满
            - strobe: 每拍内连续3次快闪，高能drop段使用
            - fire: 暖色系随机颜色循环，营造火焰氛围
            - chase_clear: 单点清屏移动，每拍CLEAR后点亮下一颗，干净追逐感
            - cylon: 红色单点来回弹跳（骑士车灯效果）
            - heartbeat: 每拍双脉冲（强-弱），心跳律动感
            - disco: 每拍4次快速全带换色，迪斯科效果
            - rainbow_pulse: 全带彩虹色按BPM推进色相，每拍跳30°
        :param duration: 时长（秒）
        """
        await self._stop_current_task()
        self._current_task = asyncio.create_task(self._bpm_flash_task(bpm, mode, duration))

    async def _bpm_flash_task(self, bpm: int, mode: str, duration: float):
        beat_interval = 60.0 / bpm
        start = time.time()

        # 4路流星头位置
        n = self._led_count
        spacing = n // 4
        meteor_pos = [i * spacing for i in range(4)]

        # 2路彩球位置（从两端出发）
        bounce_pos = [0, n - 1]
        bounce_dir = [1, -1]

        # cylon 单点来回弹跳
        cylon_pos = 0
        cylon_dir = 1

        try:
            beat_count = 0
            while time.time() - start < duration:
                beat_count += 1
                beat_start = time.time()

                if mode == "all":
                    r1, g1, b1 = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
                    await self.send_command(f"ALL {r1} {g1} {b1}")

                elif mode == "alter":
                    # 每拍全体换一个随机色（比原来55条LED命令可靠）
                    r1, g1, b1 = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
                    await self.send_command(f"ALL {r1} {g1} {b1}")

                elif mode == "running":
                    # 原始实现：每拍点亮下一颗，积累式跑马
                    pos = (beat_count - 1) % n
                    r1, g1, b1 = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
                    await self.send_command(f"LED {pos} {r1} {g1} {b1}")

                elif mode == "gradient":
                    r1, g1, b1 = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
                    brightness = 0.5 + 0.5 * (beat_count % 2)
                    await self.send_command(f"ALL {int(r1*brightness)} {int(g1*brightness)} {int(b1*brightness)}")

                elif mode == "meteor":
                    # 4路流星：每拍各移动3格，每路只发1条LED命令（共4条）
                    r1, g1, b1 = random.randint(100, 255), random.randint(100, 255), random.randint(100, 255)
                    for i in range(4):
                        meteor_pos[i] = (meteor_pos[i] + 3) % n
                        await self.send_command(f"LED {meteor_pos[i]} {r1} {g1} {b1}")

                elif mode == "sparkle":
                    # 每拍随机点亮3颗
                    for pos in random.sample(range(n), 3):
                        r, g, b = random.randint(150, 255), random.randint(150, 255), random.randint(150, 255)
                        await self.send_command(f"LED {pos} {r} {g} {b}")

                elif mode == "bounce":
                    # 2路彩球从两端向中间移动，共2条LED命令
                    step = max(1, bpm // 60)
                    r1, g1, b1 = random.randint(100, 255), random.randint(100, 255), random.randint(100, 255)
                    for idx in range(2):
                        await self.send_command(f"LED {bounce_pos[idx]} {r1} {g1} {b1}")
                        new_pos = bounce_pos[idx] + bounce_dir[idx] * step
                        if new_pos >= n:
                            new_pos = n - 1
                            bounce_dir[idx] = -1
                        elif new_pos < 0:
                            new_pos = 0
                            bounce_dir[idx] = 1
                        bounce_pos[idx] = new_pos

                elif mode == "strobe":
                    r1, g1, b1 = random.randint(150, 255), random.randint(150, 255), random.randint(150, 255)
                    flash_interval = beat_interval / 8
                    for _ in range(3):
                        await self.send_command(f"ALL {r1} {g1} {b1}")
                        await asyncio.sleep(flash_interval)
                        await self.send_command("CLEAR")
                        await asyncio.sleep(flash_interval)
                    elapsed = time.time() - beat_start
                    remaining = beat_interval - elapsed
                    if remaining > 0:
                        await asyncio.sleep(remaining)
                    continue

                elif mode == "fire":
                    # 暖色系颜色用ALL命令循环，不用单颗LED
                    warm_colors = [
                        (255, random.randint(80, 160), 0),
                        (255, random.randint(40, 80), 0),
                        (random.randint(200, 255), random.randint(20, 50), 0),
                        (random.randint(150, 220), 0, 0),
                    ]
                    r, g, b = warm_colors[beat_count % len(warm_colors)]
                    await self.send_command(f"ALL {r} {g} {b}")

                elif mode == "chase_clear":
                    # CLEAR + 1 LED：单点清屏移动，干净追逐感（不积累）
                    pos = (beat_count - 1) % n
                    r1, g1, b1 = random.randint(100, 255), random.randint(100, 255), random.randint(100, 255)
                    await self.send_command("CLEAR")
                    await self.send_command(f"LED {pos} {r1} {g1} {b1}")

                elif mode == "cylon":
                    # CLEAR + 1 LED：红色单点来回弹跳（骑士车灯）
                    await self.send_command("CLEAR")
                    await self.send_command(f"LED {cylon_pos} 255 0 0")
                    cylon_pos += cylon_dir
                    if cylon_pos >= n:
                        cylon_pos = n - 2
                        cylon_dir = -1
                    elif cylon_pos < 0:
                        cylon_pos = 1
                        cylon_dir = 1

                elif mode == "heartbeat":
                    # 双脉冲：强（full）→ 弱（half），像心跳
                    r1, g1, b1 = random.randint(150, 255), 0, 0
                    lub = beat_interval * 0.12
                    gap = beat_interval * 0.08
                    await self.send_command(f"ALL {r1} {g1} {b1}")
                    await asyncio.sleep(lub)
                    await self.send_command("CLEAR")
                    await asyncio.sleep(gap)
                    await self.send_command(f"ALL {r1 // 2} 0 0")
                    await asyncio.sleep(lub)
                    await self.send_command("CLEAR")
                    elapsed = time.time() - beat_start
                    remaining = beat_interval - elapsed
                    if remaining > 0:
                        await asyncio.sleep(remaining)
                    continue

                elif mode == "disco":
                    # 每拍4次快速全带换色
                    sub = beat_interval / 4
                    for _ in range(4):
                        r, g, b = random.randint(100, 255), random.randint(100, 255), random.randint(100, 255)
                        await self.send_command(f"ALL {r} {g} {b}")
                        await asyncio.sleep(sub)
                    continue

                elif mode == "rainbow_pulse":
                    # 全带彩虹色按BPM推进，每拍跳30°
                    hue = (beat_count * 30) % 360
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
                    await self.send_command(f"ALL {r} {g} {b}")

                elapsed = time.time() - beat_start
                remaining = beat_interval - elapsed
                if remaining > 0:
                    await asyncio.sleep(remaining)

        except Exception as e:
            print(f"BPM效果异常: {e}")
        finally:
            await self.clear()