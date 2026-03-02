"""
Cerebellum 日志配置中心

使用 loguru 提供统一的日志管理：
- 控制台输出：简洁清晰，显示关键执行状态
- 文件输出：详细完整，便于调试和问题排查
"""

import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

from loguru import logger


LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)


class InterceptHandler(logging.Handler):
    """
    拦截标准 logging 日志，转发到 loguru
    
    用于捕获第三方库的日志输出
    """
    
    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1
        
        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def setup_logging(
    debug: bool = False,
    log_file: Optional[Path] = None,
    rotation: str = "10 MB",
    retention: str = "7 days"
) -> None:
    """
    配置日志系统
    
    Args:
        debug: 是否启用调试模式
        log_file: 日志文件路径，默认自动生成
        rotation: 日志文件轮转大小
        retention: 日志文件保留时间
    """
    logger.remove()
    
    console_format = (
        "<green>{time:HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<level>{message}</level>"
    )
    
    console_level = "DEBUG" if debug else "INFO"
    logger.add(
        sys.stdout,
        format=console_format,
        level=console_level,
        colorize=True,
        backtrace=False,
        diagnose=False
    )
    
    if log_file is None:
        log_file = LOG_DIR / f"cerebellum_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    file_format = (
        "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
        "{level: <8} | "
        "{name}:{function}:{line} - "
        "{message}"
    )
    
    logger.add(
        log_file,
        format=file_format,
        level="DEBUG",
        rotation=rotation,
        retention=retention,
        compression="zip",
        encoding="utf-8",
        backtrace=True,
        diagnose=True
    )
    
    logging.basicConfig(handlers=[InterceptHandler()], level=logging.ERROR)
    
    for logger_name in [
        "httpx", "httpcore", "openai", "urllib3", "asyncio", "werkzeug",
        "deepagents", "langchain", "langgraph", "langsmith"
    ]:
        logging.getLogger(logger_name).setLevel(logging.ERROR)


def get_logger(name: str = "cerebellum"):
    """
    获取指定名称的 logger
    
    Args:
        name: 模块名称
    
    Returns:
        配置好的 logger 实例
    """
    return logger.bind(name=name)


def log_function_call(func_name: str, **kwargs):
    """
    记录函数调用日志
    
    Args:
        func_name: 函数名称
        **kwargs: 函数参数
    """
    params = ", ".join(f"{k}={v!r}" for k, v in kwargs.items() if v is not None)
    logger.debug(f"调用 {func_name}({params})")


def log_execution_time(func_name: str, elapsed_ms: int):
    """
    记录执行时间
    
    Args:
        func_name: 函数名称
        elapsed_ms: 执行时间（毫秒）
    """
    if elapsed_ms > 1000:
        logger.info(f"{func_name} 执行完成，耗时 {elapsed_ms / 1000:.2f} 秒")
    else:
        logger.debug(f"{func_name} 执行完成，耗时 {elapsed_ms} 毫秒")
