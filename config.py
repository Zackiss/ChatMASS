import torch
import yaml


class Config:
    """"""
    def __init__(self, config_path):
        self.device = None
        self.config_file = open(config_path, encoding="UTF-8")
        self._assign_config()
        # determine the device we'll train on
        if self.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def _assign_config(self):
        names = self.__dict__
        y = yaml.load(self.config_file, Loader=yaml.FullLoader)
        for _name, _value in y.items():
            names[_name] = _value
        # the followings parameters are assigned automatically,
        # just for reference at here.
        self.dim_model = 0
        self.vocab_size = 0

