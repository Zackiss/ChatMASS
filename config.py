import configparser
import os


class Config:
    def __init__(self):
        # number of blocks apply in model
        self.layer_num = self.get_config("model", "layer_num")
        # number of heads in multi-head self attention
        self.head_num = self.get_config("model", "head_num")
        # dimension of embedded word vector
        self.embedding_dim = self.get_config("model", "embedding_dim")

        """dropout setup"""
        self.attention_drop = self.get_config("dropout", "attention")
        self.embedding_drop = self.get_config("dropout", "embedding")
        self.resid_drop = self.get_config("dropout", "resid")

        """vocabulary size setup"""
        self.vocab_size = self.get_config("vocab", "vocab_size")
        self.block_size = self.get_config("vocab", "block_size")

        """trainer setup"""
        self.device = self.get_config("train", "device")
        # dataloader parameters
        self.num_workers = self.get_config("train", "num_workers")
        # optimizer parameters
        self.max_iters = self.get_config("train", "max_iters")
        self.batch_size = self.get_config("train", "batch_size")
        self.learning_rate = self.get_config("train", "learning_rate")
        self.betas = self.get_config("train", "betas")
        # only applied on matmul weights
        self.weight_decay = self.get_config("train", "weight_decay")
        self.grad_norm_clip = self.get_config("train", "grad_norm_clip")

    @staticmethod
    def get_config(section, option):
        conf = configparser.ConfigParser()
        conf.read(os.path.join(os.path.split(os.path.realpath(__file__))[0], "confug"))
        return conf.get(section, option)

