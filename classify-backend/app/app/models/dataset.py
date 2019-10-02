from datetime import datetime
from typing import List

from pydantic import BaseModel


# Shared properties
class DatasetBase(BaseModel):
    name: str = None
    description: str = None
    base_dir: str = None
    thumbnail_dir: str = None
    mask_dir: str = None


class Dataset(DatasetBase):
    id: int = None
    created: datetime = None

    class Config:
        orm_mode = True


# Properties to receive on item creation
class DatasetCreate(DatasetBase):
    pass


# Properties to receive on item update
class DatasetUpdate(DatasetBase):
    pass


class DatasetFileBase(BaseModel):
    name: str = None
    dataset_id: int = None


class DatasetFile(BaseModel):
    id: int = None
    image: str = None
    thumbnail: str = None
    meta: dict = {}
#    patientInfo: dict = None

    class Config:
        orm_mode = True


class DatasetFileCreate(DatasetFileBase):
    path: str = None
    thumbnail: str = None


class DatasetModel(BaseModel):
    id: int = None
    name: str = None


class DatasetFiles(BaseModel):
    datasets: List[Dataset] = None
    models: List[DatasetModel] = None
    items: List[DatasetFile] = None


class AnalysisModelIn(BaseModel):
    name: str = None
    dataset_id: int = None


class AnalysisModel(AnalysisModelIn):
    id: int = None
    index: str = None
