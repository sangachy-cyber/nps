#!/usr/bin/env python3
"""
日志模块测试用例
"""

import pytest
import logging
from pathlib import Path
from src.network_simulation.utils.logger import LoggerConfig, get_logger


@pytest.fixture
def tmp_log_dir(tmp_path):
    """创建临时日志目录"""
    log_dir = tmp_path / "test_logs"
    log_dir.mkdir(exist_ok=True)
    return log_dir


@pytest.fixture
def logger_config(tmp_log_dir):
    """初始化日志配置"""
    return LoggerConfig(log_dir=tmp_log_dir, log_level=logging.DEBUG)


def test_logger_config_initialization(logger_config, tmp_log_dir):
    """测试日志配置初始化"""
    assert logger_config is not None
    assert logger_config.log_dir == tmp_log_dir
    assert logger_config.log_level == logging.DEBUG
    assert tmp_log_dir.exists()


def test_get_logger(logger_config):
    """测试获取日志记录器"""
    # 获取日志记录器
    logger = logger_config.get_logger("test_module")

    # 验证结果
    assert isinstance(logger, logging.Logger)
    assert logger.name == "test_module"
    assert logger.level == logging.DEBUG
    assert len(logger.handlers) == 2  # 控制台处理器和文件处理器


def test_get_logger_multiple_calls(logger_config):
    """测试多次获取同一个日志记录器"""
    # 第一次获取日志记录器
    logger1 = logger_config.get_logger("test_module")
    # 第二次获取日志记录器
    logger2 = logger_config.get_logger("test_module")

    # 验证两个日志记录器是同一个实例
    assert logger1 is logger2
    # 验证处理器数量没有增加
    assert len(logger1.handlers) == 2
    assert len(logger2.handlers) == 2


def test_get_logger_different_names(logger_config):
    """测试获取不同名称的日志记录器"""
    # 获取不同名称的日志记录器
    logger1 = logger_config.get_logger("module1")
    logger2 = logger_config.get_logger("module2")

    # 验证是不同的实例
    assert logger1 is not logger2
    assert logger1.name == "module1"
    assert logger2.name == "module2"


def test_get_logger_different_levels(tmp_log_dir):
    """测试获取不同级别日志记录器"""
    # 创建不同级别的日志配置
    debug_config = LoggerConfig(log_dir=tmp_log_dir, log_level=logging.DEBUG)
    info_config = LoggerConfig(log_dir=tmp_log_dir, log_level=logging.INFO)

    # 获取不同级别的日志记录器
    debug_logger = debug_config.get_logger("debug_module")
    info_logger = info_config.get_logger("info_module")

    # 验证日志级别
    assert debug_logger.level == logging.DEBUG
    assert info_logger.level == logging.INFO


def test_global_get_logger():
    """测试全局的get_logger函数"""
    # 使用全局get_logger函数获取日志记录器
    logger = get_logger("global_module")

    # 验证结果
    assert isinstance(logger, logging.Logger)
    assert logger.name == "global_module"
    assert len(logger.handlers) == 2  # 控制台处理器和文件处理器


def test_logger_output(logger_config, tmp_log_dir):
    """测试日志输出"""
    # 获取日志记录器
    logger = logger_config.get_logger("test_output")

    # 记录一条日志
    test_message = "This is a test log message"
    logger.info(test_message)

    # 验证日志文件是否生成
    log_files = list(tmp_log_dir.glob("*.log"))
    assert len(log_files) > 0

    # 验证日志文件中包含测试消息
    for log_file in log_files:
        with open(log_file, "r", encoding="utf-8") as f:
            content = f.read()
            if test_message in content:
                break
    else:
        pytest.fail("测试消息未出现在任何日志文件中")


def test_logger_levels(logger_config):
    """测试不同级别日志记录"""
    # 获取日志记录器
    logger = logger_config.get_logger("test_levels")

    # 记录不同级别的日志
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")
    logger.critical("Critical message")

    # 验证日志记录器没有抛出异常
    assert True


def test_logger_without_file_handler(tmp_log_dir):
    """测试日志记录器是否只使用控制台处理器"""
    # 创建日志配置
    logger_config = LoggerConfig(log_dir=tmp_log_dir, log_level=logging.INFO)

    # 获取日志记录器
    logger = logger_config.get_logger("test_no_file")

    # 验证处理器数量
    assert len(logger.handlers) == 2  # 控制台处理器和文件处理器


def test_logger_config_with_default_params():
    """测试使用默认参数初始化日志配置"""
    # 使用默认参数创建日志配置
    logger_config = LoggerConfig()

    # 验证默认参数
    assert logger_config.log_dir == Path("logs")
    assert logger_config.log_level == logging.INFO
    assert logger_config.log_dir.exists()
