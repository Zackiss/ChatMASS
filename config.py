import configparser
import os


class Config:
    def __init__(self):
        # number of blocks apply in model
        self.layer_num = int(self.get_config("model", "layer_num"))
        # number of heads in multi-head self attention
        self.head_num = int(self.get_config("model", "head_num"))
        # dimension of embedded word vector
        self.embedding_dim = int(self.get_config("model", "embedding_dim"))
        self.dim_model = 0
        self.tokenization = bool(self.get_config("model", "tokenization"))
        self.token_batch = int(self.get_config("model", "token_batch"))

        """dropout setup"""
        self.attention_drop = float(self.get_config("dropout", "attention"))
        self.embedding_drop = float(self.get_config("dropout", "embedding"))
        self.resid_drop = float(self.get_config("dropout", "resid"))

        """vocabulary size setup"""
        self.vocab_size = 0
        self.block_size = int(self.get_config("vocab", "block_size"))

        """trainer setup"""
        self.device = self.get_config("train", "device")
        # dataloader parameters
        self.num_workers = int(self.get_config("train", "num_workers"))
        # optimizer parameters
        self.max_iters = int(self.get_config("train", "max_iters"))
        self.batch_size = int(self.get_config("train", "batch_size"))
        self.learning_rate = float(self.get_config("train", "learning_rate"))
        # only applied on matmul weights
        self.weight_decay = float(self.get_config("train", "weight_decay"))
        self.grad_norm_clip = float(self.get_config("train", "grad_norm_clip"))
        self.gradient_accumulation_steps = int(self.get_config("train", "gradient_accumulation_steps"))

    @staticmethod
    def get_config(section, option):
        conf = configparser.ConfigParser()
        conf.read("config.ini")
        return conf.get(section, option)

