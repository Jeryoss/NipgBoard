import logging


def setup_logger(log_file_name=None, include_time=False, include_date=False):
    """
    Set up a logger object that writes log messages to a file.

    Args:
        log_file_name (str, optional): The name of the file where log messages will be written. If not specified, log messages will be printed to the console.
        include_time (bool, optional): If True, includes the time in the log messages. Default is False.
        include_date (bool, optional): If True, includes the date in the log messages. Default is False.

    Returns:
        logging.Logger: A logger object that can be used to write log messages.
    """
    # Create a logger object with the desired name
    logger = logging.getLogger("my_logger")

    # Set the log level to INFO, which will capture messages with severity level INFO and above
    logger.setLevel(logging.INFO)

    # Create a formatter for the log messages
    formatter_string = '%(asctime)s '
    if include_time:
        formatter_string += '%(asctime)s '
    if include_date:
        formatter_string += '%(asctime)s '
    formatter_string += '%(levelname)s: %(message)s'
    formatter = logging.Formatter(formatter_string)

    # If a logfile name is specified, create a file handler that writes log messages to the file
    if log_file_name:
        file_handler = logging.FileHandler(log_file_name)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    # If no logfile name is specified, create a console handler that prints log messages to the console
    else:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger
