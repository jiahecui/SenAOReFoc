import logging
module_name = __name__.split('.')[0]
logger = logging.getLogger(module_name)

def get_logger(name=module_name):
    return logging.getLogger(name)