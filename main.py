import json
import os
import time

import hanlp
import torch
from config import Config
from model import Model
from torch.utils.data import Dataset
from train import Trainer
import numpy as np


class ChatDataset(Dataset):
    """"""
    def __init__(self, config):
        self.config = config
        self.dictionary = {}
        self.tokenized_text = []
        self.eliminated_text = []
        self.block_size = self.config.block_size
        if self.config.tokenization:
            self.tok_fine = hanlp.load(hanlp.pretrained.tok.FINE_ELECTRA_SMALL_ZH)
            self.sts = hanlp.load(hanlp.pretrained.sts.STS_ELECTRA_BASE_ZH)
        self.data = self.load_data()
        print("dict sample check: " + str(self.dictionary.keys[15:25]))
        print("data sample check: " + str(self.data[15:25]))
        self.chars = sorted(list(set(self.data)))
        self.data_size, self.vocab_size = len(self.data), self.chars[-1]
        # self.stoi = {ch: j for j, ch in enumerate(self.chars)}
        # self.itos = {j: ch for j, ch in enumerate(self.chars)}

    def __getitem__(self, index):
        chat = self.data[index: index + self.block_size + 1]
        x = torch.tensor(chat[:-1], dtype=torch.long)
        y = torch.tensor(chat[1:], dtype=torch.long)
        return x, y

    def __len__(self):
        return len(self.data) - self.block_size

    def load_data(self) -> list[int]:
        """"""
        cur_data = []
        whole_sentence = ""
        # clean: 36, 61, 86, 111, 136, 161, 361
        # 61, 86, 111, 136, 161, 186,
        # 211, 236, 261, 286, 361
        for name in [36]:
            with open('Trainset/art{0}.json'.format(name), 'r') as f:
                print('reading art{0}.json'.format(name))
                score = json.load(f)
            cur_data += score
        for sentence in cur_data:
            whole_sentence += sentence
        self.generate_dict(whole_sentence)
        processed_sentence = self.sentence_parse_to_ids(whole_sentence)
        return processed_sentence

    def idx_to_words(self, vector: np.array) -> list[str]:
        """"""
        reply = []
        for index in vector:
            reply.append(self.eliminated_text[index])
        return reply

    def generate_dict(self, whole_sentence: str):
        if self.config.tokenization:
            split = whole_sentence.split("。", maxsplit=self.config.token_batch)
            whole_sentence_tokenized = []
            for batch in split:
                self.tokenized_text = self.tok_fine(batch)
                whole_sentence_tokenized += self.tokenized_text
            self.tokenized_text = whole_sentence_tokenized
        else:
            self.tokenized_text = list(whole_sentence)
        self.eliminated_text = list(set(self.tokenized_text))
        index = 0
        for w in self.eliminated_text:
            self.dictionary[w] = index
            index += 1

    def sentence_parse_to_ids(self, sentence: str, ambiguity=False) -> list[int]:
        """return the embedding matrix (word_dim) x (word_count)"""
        if self.config.tokenization:
            self.tokenized_text = self.tok_fine(sentence)
            if ambiguity:
                specified_text = []
                for w in self.tokenized_text:
                    # replace the unknown word with most similar word from dictionary
                    if w not in self.eliminated_text:
                        sim_tuples = []
                        for val in enumerate(self.eliminated_text):
                            sim_tuples.append((w, val))
                        sim_res = self.sts(sim_tuples)
                        # find the most similar word
                        r = self.eliminated_text[sim_res.index(max(sim_res))]
                        print(w + " replaced  to" + r)
                        w = r
                    specified_text.append(w)
                self.tokenized_text = specified_text
        indices_text = [self.dictionary[w] for w in self.tokenized_text]
        return indices_text


if __name__ == "__main__":
    """basic model setup"""
    conf = Config()

    dataset = ChatDataset(conf)
    conf.vocab_size = dataset.vocab_size + 1
    conf.embedding_dim = int(conf.embedding_dim / conf.head_num) * conf.head_num
    conf.dim_model = conf.embedding_dim
    print("working with vocabulary size: " + str(conf.vocab_size) + ", " +
          "embedding dimension: " + str(conf.dim_model))

    model = Model(conf)

    if os.path.exists("trained_model"):
        model.load_state_dict(torch.load("trained_model"))
        model.eval()
    else:
        trainer = Trainer(conf, model=model, train_dataset=dataset)
        trainer.run()

    query = "我早上醒来"
    x = torch.tensor([dataset.sentence_parse_to_ids(query, ambiguity=True)], dtype=torch.long).to("cuda")
    y = model.generate(x, 500, temperature=0.6, top_k=35)[0]
    completion = "".join(dataset.idx_to_words(y))
    for word in completion:
        print(word, end="")
        time.sleep(0.1)
