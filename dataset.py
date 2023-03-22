import os

import hanlp
import jieba
from torch.utils.data import Dataset
import torch
import numpy as np
import logging
import pyarrow.parquet as pq


class ChatDataset(Dataset):
    """"""
    def __init__(self, file_paths: list[str], config):
        self.config = config

        """database setup"""
        # engine = create_engine(f'sqlite:///{file_paths}?check_same_thread=False', echo=True)
        # DBSession = sessionmaker(bind=engine)
        # self.dialogue_table = Dialogue()
        # self.session = DBSession()
        # self.session.add_all([self.dialogue_table])

        con = False
        iter_num = 0
        """parquet setup"""
        for file_path in file_paths:
            if iter_num < config.max_trainset_num:
                df = pq.read_pandas(file_path).to_pandas()
                # [np.array, np.array,...]
                self.data = np.concatenate((self.data, df.values), axis=0) if con else df.values
                con = True if not con else True
                print(f"data loaded in size of {self.data.size}")
                iter_num += 1
        logging.debug(f"data loaded done with iter {iter_num}, size {self.data.size}")

        """dataset initialize"""
        self.dictionary = {}
        self.vocab_list = []
        self.eliminated_text = []
        self.block_size = self.config.block_size
        self.data_size = self.data.size
        if self.config.tokenization:
            self.tok_fine = hanlp.load(hanlp.pretrained.tok.FINE_ELECTRA_SMALL_ZH)
            self.sts = hanlp.load(hanlp.pretrained.sts.STS_ELECTRA_BASE_ZH)
        self.generate_dict()
        self.vocab_size = len(self.vocab_list)
        logging.debug("dict sample check: " + str(list(self.dictionary.keys())[15:25]))
        logging.debug("data size: " + str(self.data_size))
        logging.debug("vocab size: " + str(self.vocab_size))

    def __getitem__(self, index: int):
        # Use the select function to get the target value with the specified id
        # stmt = select(Dialogue.target).where(Dialogue.id == index+1)
        # dialog = str(self.session.execute(stmt).scalar())

        dialog = str(list(self.data[index, :])[0])
        while len(dialog) < self.config.block_size+1:
            dialog += dialog
        chat = dialog[:self.block_size+1]
        chat = self.sentence_parse_to_ids(chat)
        logging.debug(f"fetch with index {index}, content: " + str(chat))
        x = torch.tensor(chat[1:], dtype=torch.long)
        y = torch.tensor(chat[:-1], dtype=torch.long)
        return x, y

    def __len__(self):
        return self.data_size

    def idx_to_words(self, vector: np.array) -> list[str]:
        """
        recover words back from indices to recognizable human words
        """
        reply = []
        for index in vector:
            reply.append(self.vocab_list[index])
        return reply

    def generate_dict(self):
        """
        procedure of generating the vocab dictionary with the given train set for forward labeling
        simply use eliminated text list when finding words' index is more efficient
        """
        # stmt = select(Dialogue.target)
        # Execute the statement and fetch all results
        # dialogs = self.session.execute(stmt).fetchall()
        logging.debug("sentence tokenization start")
        self.eliminated_text = set()
        if self.config.tokenization:
            if self.config.deepl_tokenize:
                # use transformer to tokenize sentences
                for d in np.nditer(self.data, flags=['refs_ok']):
                    self.eliminated_text.union(self.eliminated_text, set(self.tok_fine(str(d))))
                    if len(self.eliminated_text) >= self.config.max_vocab_size:
                        break
            else:
                # use normal word cutting tool based on HMM to tokenize sentences
                for d in np.nditer(self.data, flags=['refs_ok']):
                    self.eliminated_text.union(self.eliminated_text, set(jieba.cut(str(d))))
                    if len(self.eliminated_text) >= self.config.max_vocab_size:
                        break
        else:
            if self.config.use_custom_dict:
                # skip the tokenized period, generate dictionary by given dictionary
                pass
            else:
                # normal way to cut all words into set with minimum unit of single word
                for d in np.nditer(self.data, flags=['refs_ok']):
                    self.eliminated_text = set.union(self.eliminated_text, set(list(str(d))))
                    if len(self.eliminated_text) >= self.config.max_vocab_size:
                        break

        logging.debug("sentence tokenized and eliminated")
        if not self.config.use_custom_dict:
            index = 0
            for word in self.eliminated_text:
                self.dictionary[word] = index
                self.vocab_list.append(word)
                index += 1
        else:
            with open("dict/" + os.listdir("dict")[0], 'r') as f:
                for idx, line in enumerate(f):
                    word = line.strip()
                    self.dictionary[word] = idx
                    self.vocab_list.append(word)
        logging.debug("dictionary created")

    def sentence_parse_to_ids(self, sentence: str, ambiguity=False) -> list[int]:
        """
        cut the sentence to words if applicable, embed words to indices
        ambiguity for finding similar words from dictionary, applicable when parsing user input with small dict
        return the embedding matrix (word_dim) x (word_count)
        """
        if self.config.tokenization:
            tokenized_text = self.tok_fine(sentence)
            if ambiguity:
                specified_text = []
                for w in tokenized_text:
                    # replace the unknown word with most similar word from dictionary
                    if w not in self.vocab_list:
                        sim_tuples = []
                        for val in enumerate(self.eliminated_text):
                            sim_tuples.append((w, val))
                        sim_res = self.sts(sim_tuples)
                        # find the most similar word
                        r = self.eliminated_text[sim_res.index(max(sim_res))]
                        logging.info(w + " replaced  to" + r)
                        w = r
                    specified_text.append(w)
                tokenized_text = specified_text
            indices_text = [self.dictionary[w] for w in tokenized_text]
        else:
            indices_text = []
            for w in sentence:
                try:
                    cart = self.dictionary[w]
                except KeyError:
                    cart = 0
                indices_text.append(cart)
        return indices_text
