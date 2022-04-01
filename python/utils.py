
def load_logger():
    import logging
    import logging.config
    import yaml

    with open('logging.yaml', 'r') as f:
        config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)
