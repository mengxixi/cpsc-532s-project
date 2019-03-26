import yaml

class Config(object):
    CONFIG_DICT = {}
    LOADED = False

    @staticmethod
    def load_config(conf='config.yaml'):
        if not Config.LOADED:
            with open(conf, 'r') as config_file:
                Config.CONFIG_DICT = yaml.load(config_file.read())
            print("Loaded config: %s" % Config.CONFIG_DICT)
            Config.LOADED = True

    @staticmethod
    def get(key, default=None):
        parts = key.split(".")
        conf = Config.CONFIG_DICT
        for part in parts:
            conf = conf.get(part, None)
            # At any level, if the return is None, return default
            if conf is None:
                return default

        # Got to the last level
        return conf
