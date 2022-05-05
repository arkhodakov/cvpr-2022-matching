import logging
import logging.config
import yaml


def load_logger() -> None:
    """Open default logging configuration and use it as default config for root logger."""
    with open('logging.yaml', 'r') as f:
        config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)
