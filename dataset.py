import hanlp
from database import Dialogue, Emotion
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from torch.utils.data import Dataset
import torch
import numpy as np
import logging


class ChatDataset(Dataset):
    """"""
    def __init__(self, file_paths: list[str], config):
        self.config = config

        """database setup"""
        password, port = "password", 3306
        engine = create_engine(f'mysql+mysqlconnector://root:{password}@localhost:{port}/test')
        DBSession = sessionmaker(bind=engine)
        # initialize tables of train sets
        self.dialogue_table = Dialogue(config=config)
        self.emotion_table = Emotion(config=config)
        # create session
        self.session = DBSession()
        self.session.add_all([self.dialogue_table, self.emotion_table])

        """dataset initialize"""
        self.dictionary = {}
        self.tokenized_text = []
        self.eliminated_text = []
        self.block_size = self.config.block_size
        if self.config.tokenization:
            self.tok_fine = hanlp.load(hanlp.pretrained.tok.FINE_ELECTRA_SMALL_ZH)
            self.sts = hanlp.load(hanlp.pretrained.sts.STS_ELECTRA_BASE_ZH)
        self.generate_dict()
        logging.debug("dict sample check: " + str(list(self.dictionary.keys())[15:25]))
        self.data_size, self.vocab_size = len(self.tokenized_text), len(self.eliminated_text)

    def __getitem__(self, index: int):
        # Example on fetch one certain dialog with ids
        dialog = self.session.query(self.dialogue_table).filter(self.dialogue_table.id == index).first()
        logging.debug("data fetch: " + str(dialog))
        chat = dialog[index: index + self.block_size + 1]
        x = torch.tensor(chat[:-1], dtype=torch.long)
        y = torch.tensor(chat[1:], dtype=torch.long)
        return x, y

    def __len__(self):
        return len(self.data) - self.block_size

    def idx_to_words(self, vector: np.array) -> list[str]:
        """
        recover words back from indices to recognizable human words
        """
        reply = []
        for index in vector:
            reply.append(self.eliminated_text[index])
        return reply

    def generate_dict(self):
        """
        procedure of generating the vocab dictionary with the given train set for forward labeling
        simply use eliminated text list when finding words' index is more efficient
        """
        dialogs = self.session.query(self.dialogue_table).all()
        # Concatenate the dialog content into a single string
        whole_sentence = ''.join([d.content for d in dialogs])

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
        """
        cut the sentence to words if applicable, embed words to indices
        ambiguity for finding similar words from dictionary, applicable when parsing user input with small dict
        return the embedding matrix (word_dim) x (word_count)
        """
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
                        logging.info(w + " replaced  to" + r)
                        w = r
                    specified_text.append(w)
                self.tokenized_text = specified_text
            indices_text = [self.dictionary[w] for w in self.tokenized_text]
        else:
            indices_text = [self.dictionary[w] for w in sentence]
        return indices_text
