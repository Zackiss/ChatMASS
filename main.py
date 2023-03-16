import os
import time

import torch
from config import Config
from model import Model
from train import Trainer
from dataset import ChatDataset
import logging

if __name__ == "__main__":
    """config init"""
    logging.basicConfig(filename="report.log", level=logging.DEBUG)
    conf = Config(r"config.yaml")

    """basic model setup"""
    dataset = ChatDataset([], conf)
    conf.vocab_size = dataset.vocab_size
    conf.embedding_dim = int(conf.embedding_dim / conf.head_num) * conf.head_num
    conf.dim_model = conf.embedding_dim
    logging.info("working with vocabulary size: " + str(conf.vocab_size) + ", " +
                 "embedding dimension: " + str(conf.dim_model))

    model = Model(conf)

    if os.path.exists("trained_model"):
        model.load_state_dict(torch.load("trained_model"))
        model.eval()
    else:
        trainer = Trainer(conf, model=model, train_dataset=dataset)
        trainer.run()

    query = "我早上醒来"
    x = torch.tensor([dataset.sentence_parse_to_ids(query, ambiguity=True)], dtype=torch.long).to(conf.device)
    y = model.generate(x, 500, temperature=0.6, top_k=35)[0]
    completion = "".join(dataset.idx_to_words(y))
    for word in completion:
        print(word, end="")
        time.sleep(0.1)
    dataset.session.close()
