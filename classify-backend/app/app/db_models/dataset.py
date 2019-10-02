import datetime

from sqlalchemy import Column, ForeignKey, Integer, String, DateTime
from sqlalchemy.orm import relationship

from app.db.base_class import Base


class Dataset(Base):
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    description = Column(String, default="")
    base_dir = Column(String)
    thumbnail_dir = Column(String, default="")
    mask_dir = Column(String, default="")
    created = Column(DateTime, default=datetime.datetime.utcnow)
    model = relationship("AnalysisModel", back_populates="dataset")
    files = relationship("DatasetFile", back_populates="dataset")


class AnalysisModel(Base):
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    index = Column(String, default="")
    dataset_id = Column(Integer, ForeignKey("dataset.id"), index=True)
    dataset = relationship("Dataset", back_populates="model")


class DatasetFile(Base):
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    path = Column(String)
    thumbnail = Column(String)
    mask = Column(String)
    meta = Column(String)
    dataset_id = Column(Integer, ForeignKey("dataset.id"), index=True)
    dataset = relationship("Dataset", back_populates="files")