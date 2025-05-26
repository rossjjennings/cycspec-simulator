from loguru import logger
import sys

def enable(level="INFO"):
    logger.add(
        sys.stderr,
        level=level,
        format="<lvl>{level:<8}</lvl> ({name}:{line}): <lvl>{message}</lvl>"
    )

def disable():
    logger.remove()

def test():
    logger.debug("A debug message")
    logger.info("An info message")
    logger.warning("A warning message")
    logger.error("An error message")
    logger.critical("A critical message")

disable()
