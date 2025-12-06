#!/usr/bin/env python3
"""
日志配置模块

该模块提供了统一的日志配置功能，支持不同级别的日志记录，
同时输出到控制台和文件，方便后续回溯和调试。
"""

import logging
from pathlib import Path
from datetime import datetime


class LoggerConfig:
    """日志配置类"""

    def __init__(self, log_dir: Path = Path("logs"), log_level: int = logging.INFO):
        """初始化日志配置

        Args:
            log_dir: 日志文件保存目录
            log_level: 日志级别，默认为INFO
        """
        self.log_dir = log_dir
        self.log_level = log_level

        # 创建日志目录
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # 设置日志格式
        self.formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
        )

    def get_logger(self, name: str) -> logging.Logger:
        """获取指定名称的日志记录器

        Args:
            name: 日志记录器名称，通常使用__name__

        Returns:
            logging.Logger: 配置好的日志记录器
        """
        # 创建日志记录器
        logger = logging.getLogger(name)
        logger.setLevel(self.log_level)

        # 避免重复添加处理器
        if not logger.handlers:
            # 控制台处理器
            console_handler = logging.StreamHandler()
            console_handler.setLevel(self.log_level)
            console_handler.setFormatter(self.formatter)
            logger.addHandler(console_handler)

            # 文件处理器
            # 按日期和模块命名日志文件
            timestamp = datetime.now().strftime("%Y%m%d")
            module_name = name.split(".")[-1]
            log_file = self.log_dir / f"{timestamp}_{module_name}.log"

            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setLevel(self.log_level)
            file_handler.setFormatter(self.formatter)
            logger.addHandler(file_handler)

        return logger


# 创建全局日志配置实例
logger_config = LoggerConfig()


def get_logger(name: str) -> logging.Logger:
    """获取日志记录器的便捷函数

    Args:
        name: 日志记录器名称，通常使用__name__

    Returns:
        logging.Logger: 配置好的日志记录器
    """
    return logger_config.get_logger(name)
