from config import Config
from model import Model
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertModel
from train import Trainer
import numpy as np


class ChatDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

    def __getitem__(self, index):
        chat = self.data[index]
        # 处理聊天对话数据，转换为张量或序列
        processed_chat = self.process_chat(chat)
        return processed_chat

    def __len__(self):
        return len(self.data)

    def process_chat(self, chat: str):
        """"""
        processed_chat = self.sentence_parse_to_ids(chat)
        return processed_chat

    def idx_to_word(self, vector: np.array):
        """"""
        reply = self.tokenizer.decode(vector)
        return reply

    def sentence_parse_to_ids(self, sentence: str):
        """return the embedding matrix (word_dim) x (word_count)"""
        tokenized_text = self.tokenizer.tokenize(sentence, add_special_tokens=False)
        embedded_array = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        return embedded_array


if __name__ == "__main__":
    """basic model setup"""
    config = Config()
    model = Model(config)

    data = ?
    dataset = ChatDataset(data)

    trainer = Trainer(config, model=model, train_dataset=dataset)
    trainer.run()

    query = "hello world"
    reply_length = 25
    ans_vec = model.generate(dataset.sentence_parse_to_ids(query), max_new_tokens=reply_length, do_sample=True, top_k=6)
    for i in range(reply_length):
        cur_ans_vec = model.generate(ans_vec, max_new_tokens=reply_length, do_sample=True, top_k=6)
        print(dataset.idx_to_word(cur_ans_vec), end="")
        ans_vec = np.concatenate(ans_vec, cur_ans_vec)