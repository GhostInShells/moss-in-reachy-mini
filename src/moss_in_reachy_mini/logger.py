import logging
from pathlib import Path


def setup_logger(log_file: str) -> logging.Logger:
    """设置日志记录器：控制台打印Error及以上，文件写入Debug及以上"""
    # 创建日志器（核心级别设为最低的DEBUG，确保所有处理器能拿到对应日志）
    logger = logging.getLogger("moss")
    logger.setLevel(logging.DEBUG)
    # 关闭日志器的传播（避免被root logger重复处理）
    logger.propagate = False

    # 避免重复添加handler（如果已有处理器，直接返回）
    if logger.handlers:
        return logger

    # 确保日志目录存在
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # ========== 1. 文件处理器：DEBUG级别（写入所有详细日志） ==========
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)  # 文件记录DEBUG及以上
    # 文件日志格式（保留文件名、行号）
    file_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s  - %(filename)s:%(lineno)d "
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # ========== 2. 控制台处理器：ERROR级别（只打印错误日志） ==========
    console_handler = logging.StreamHandler()  # 默认输出到控制台
    console_handler.setLevel(logging.DEBUG)  # 控制台只显示ERROR及以上
    # 控制台日志格式（简化，可选）
    console_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return logger