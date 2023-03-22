import os.path
import time

import requests
from sqlalchemy import create_engine, Column, String, Integer
from sqlalchemy.orm import sessionmaker, declarative_base

engine = create_engine('sqlite:///dialogue.db?check_same_thread=False', echo=True)
Session = sessionmaker(bind=engine)
Base = declarative_base()
divide_mark = " [EOU] "

last = 0
with open("process.save", mode="r", encoding="utf-8") as process:
    last = int(process.read())


class Dialogue(Base):
    """the chart that store multi-turn dialogues"""
    __tablename__ = 'dialogue'
    # chart structure
    id = Column(Integer, primary_key=True, autoincrement=True)
    target = Column(String)


name = "dialogues_train"
directory = os.path.dirname(__file__) + "/../trainset/raw/dialogue/eng_dia/train/"

with open(directory + name + ".txt", mode="r", encoding="utf-8") as f:
    data = f.read()

session = Session()
data = data.split("\n")
result = []
header = {
    "content-length": "138",
    "content-type": "text/html; charset=UTF-8"
}
url = "https://api.niutrans.com/NiuTransServer/translation"

for i in range(last, len(data)):
    line = data[i]
    line = line.split('__eou__')

    result = ""
    for text in line:
        if text.strip() == "":
            break
        form = {
            "src_text": f"{text}",
            "from": "en",
            "to": "zh",
            "dicNo": "",
            "memoryNo": "",
            "apikey": "66141b9c945f101ab1d6479d7549078f"
        }
        _pass = True
        while _pass:
            try:
                text = eval(requests.post(url=url, headers=header, data=form).text)["tgt_text"]
                _pass = False
            except SyntaxError:
                time.sleep(2)
                _pass = True
            except KeyError:
                print(eval(requests.post(url=url, headers=header, data=form).text))
        result = result + text + divide_mark

    Final_result = Dialogue(target=result)
    Final_result.__table__.create(engine, checkfirst=True)
    session.add(Final_result)
    session.commit()
    with open("process.save", mode="w", encoding="utf-8") as process:
        process.write(str(i + 1))

session.close()

# with open(directory + 'translated'+ name +'.txt', 'w') as sf:
