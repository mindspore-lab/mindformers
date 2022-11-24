from ..tools import logger
from ..xformer_book import print_dict

class BaseConfig(dict):
    _support_list = []

    def __init__(self, **kwargs):
        super(BaseConfig, self).__init__()
        self.update(kwargs)

    def __getattr__(self, key):
        if key not in self:
            return None
        return self[key]

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        del self[key]

    def to_dict(self):
        return_dict = {}
        for key, val in self.items():
            if isinstance(val, BaseConfig):
                val = val.to_dict()
            return_dict[key] = val
        return return_dict

    @classmethod
    def show_support_list(cls):
        logger.info(f"support list of {cls.__name__} is:")
        print_dict(cls._support_list)