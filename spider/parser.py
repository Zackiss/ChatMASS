import os.path
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

for i in range(last, len(data)):
    line = data[i]
    line = line.split('__eou__')

    result = ""
    for text in line:
        if text.strip() == "":
            break
        url = "https://api.niutrans.com/"
        form = {"data": {
            "to": "zh",
            "error_code": "13002",
            "from": "en",
            "error_msg": "apikey is empty",
            "src_text": f"{text}",
            "apikey": "66141b9c945f101ab1d6479d7549078f"
        }
        }
        text = requests.post(url, data=form).content
        print(text)
        exit()
        result = result + text + divide_mark

    Final_result = Dialogue(target=result)
    Final_result.__table__.create(engine, checkfirst=True)
    session.add(Final_result)
    session.commit()
    with open("process.save", mode="w", encoding="utf-8") as process:
        process.write(str(i + 1))

session.close()

# with open(directory + 'translated'+ name +'.txt', 'w') as sf:
