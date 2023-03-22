from sqlalchemy import Column, String, Integer
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class Dialogue(Base):
    """the chart that store multi-turn dialogues"""
    __tablename__ = 'dialogue'

    # chart structure
    id = Column(Integer, primary_key=True)
    target = Column(String)


class Emotion(Base):
    """the chart that store emotional sentences"""
    __tablename__ = 'emotion'

    # chart structure
    id = Column(Integer, primary_key=True)
    target = Column(String)
    emotion = Column(Integer)
