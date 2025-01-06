import logging
import os

from colorama import Fore, Back, Style

__all__ = ['init_logger', 'debug', 'info', 'warning', 'error', 'critical']

# the name of the logger, the only requirement is that it should be unique
LOGGER_COMMON_NAME = 'THIS IS DEFINITELY A UNIQUE LOGGER NAME'


def add_coloring_to_emit_ansi(fn):
    def new(*args):
        levelno = args[0].levelno
        if levelno >= 50:  # CRITICAL / FATAL
            color = Fore.RED
        elif levelno >= 40:  # ERROR
            color = Fore.RED
        elif levelno >= 30:  # WARNING
            color = Fore.YELLOW
        elif levelno >= 20:  # INFO
            color = ''
        else:  # DEBUG
            color = Fore.BLACK + Back.WHITE
        args[0].msg = color + ' ' + args[0].msg + ' ' + Style.RESET_ALL
        return fn(*args)

    return new


def init_logger(logger_path=None, console_logger_level=logging.INFO, file_logger_level=logging.DEBUG):
    """ create a logger with name and path, output to both console and file

    :param logger_path: the path of the logger, if None, only output to console
    :param console_logger_level: the level of console logger
    :param file_logger_level: the level of file logger
    :return: logger
    """
    new_logger = logging.getLogger(LOGGER_COMMON_NAME)
    new_logger.setLevel(logging.DEBUG)

    if logger_path is not None:
        # create file handler
        os.makedirs(os.path.dirname(logger_path), exist_ok=True)
        file_handler = logging.FileHandler(logger_path)
        file_handler.setLevel(file_logger_level)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', '%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(file_formatter)
        new_logger.addHandler(file_handler)

    # create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_logger_level)
    console_formatter = logging.Formatter('%(asctime)s: %(message)s', '%H:%M:%S')
    console_handler.emit = add_coloring_to_emit_ansi(console_handler.emit)
    console_handler.setFormatter(console_formatter)
    new_logger.addHandler(console_handler)

    return new_logger


def debug(*message):
    message = ' '.join([str(m) for m in message])
    logging.getLogger(LOGGER_COMMON_NAME).debug(message)


def info(*message):
    message = ' '.join([str(m) for m in message])
    logging.getLogger(LOGGER_COMMON_NAME).info(message)


def warning(*message):
    message = ' '.join([str(m) for m in message])
    logging.getLogger(LOGGER_COMMON_NAME).warning(message)


def error(*message):
    message = ' '.join([str(m) for m in message])
    logging.getLogger(LOGGER_COMMON_NAME).error(message)


def critical(*message):
    message = ' '.join([str(m) for m in message])
    logging.getLogger(LOGGER_COMMON_NAME).critical(message)


if __name__ == '__main__':
    info('BEFORE: THIS IS AN INFO')
    debug('BEFORE: DEBUG MESSAGE: 1234.6546456')
    warning('BEFORE: THIS IS AN WARNING')
    error('BEFORE: THIS IS AN ERROR')
    init_logger()
    info('THIS IS AN INFO')
    debug('DEBUG MESSAGE: 1234.6546456')
    warning('THIS IS AN WARNING')
    error('THIS IS AN ERROR')
