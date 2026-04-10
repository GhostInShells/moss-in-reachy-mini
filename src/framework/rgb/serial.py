from serial.tools import list_ports


def auto_detect_port():
    # 获取所有可用串口
    available_ports = list_ports.comports()
    if not available_ports:
        return None

    # 常见STM32 USB转串口芯片的VID/PID
    stm32_vid_pid = [
        (0x1A86, 0x7523),  # CH340
        (0x10C4, 0xEA60),  # CP2102
        (0x0403, 0x6001),  # FT232
        (0x0483, 0x5740),  # STM32 CDC
    ]

    # 先尝试匹配已知硬件ID
    for port_info in available_ports:
        if port_info.vid is not None and port_info.pid is not None:
            for (vid, pid) in stm32_vid_pid:
                if port_info.vid == vid and port_info.pid == pid:
                    return port_info.device
    return None
