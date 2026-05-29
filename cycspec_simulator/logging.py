from loguru import logger
import sys

def enable(level="INFO", use_stdout=False):
    """
    Enable logging at a specified level.

    Parameters
    ----------
    level: str or int
        The minimum level of messages to log. Can be specified as a string
        or as an integer. The mapping (determined by loguru) is as follows:
        "DEBUG"=10, "INFO"=20, "ERROR"=30, "WARNING"=40, "CRITICAL"=50.
    use_stdout: bool, default False
        If `True`, only messages of "WARNING" level or above will be sent to
        stderr; others will be sent to stdout. Otherwise, all messages will
        be sent to stderr. Setting this to `True` is convenient in notebooks,
        where stderr output is printed with a colored background.
        In a shell, it is preferable to send all messages to stderr so that
        logging output can be redirected separately from stdout.
    """
    fmt = "<lvl>{level:<8}</lvl> ({name}:{line}): {message}"
    def stdout_filter(record):
        return record["level"].no < logger.level("WARNING").no

    if use_stdout:
        logger.add(sys.stdout, level=level, format=fmt, filter=stdout_filter)
        logger.add(sys.stderr, level="WARNING", format=fmt)
    else:
        logger.add(sys.stderr, level=level, format=fmt)

def disable():
    """
    Disable logging.
    """
    logger.remove()

def test():
    """
    Test logging a message at each level.
    """
    logger.debug("A debug message")
    logger.info("An info message")
    logger.warning("A warning message")
    logger.error("An error message")
    logger.critical("A critical message")

disable()
