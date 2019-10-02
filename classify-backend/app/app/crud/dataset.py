from typing import List, Optional

from fastapi.encoders import jsonable_encoder
from sqlalchemy.orm import Session

from app.db_models.dataset import Dataset as DBDataset, DatasetFile as DBDatasetFile
from app.models.dataset import DatasetCreate, DatasetUpdate


def get(db_session: Session, *, id: int) -> Optional[DBDataset]:
    return db_session.query(DBDataset).filter(DBDataset.id == id).first()


def create(db_session: Session, *, dataset_in: DatasetCreate) -> DBDataset:
    item_in_data = jsonable_encoder(dataset_in)
    item = DBDataset(**item_in_data)
    db_session.add(item)
    db_session.commit()
    db_session.refresh(item)
    return item


def update(db_session: Session, *, dataset: DBDataset, dataset_in: DatasetUpdate) -> DBDataset:
    dataset_data = jsonable_encoder(dataset)
    update_data = dataset_in.dict(skip_defaults=True)
    for field in dataset_data:
        if field in update_data:
            setattr(dataset, field, update_data[field])
    db_session.add(dataset)
    db_session.commit()
    db_session.refresh(dataset)
    return dataset


def remove(db_session: Session, *, id: int):
    dataset = db_session.query(DBDataset).filter(DBDataset.id == id).first()
    db_session.query(DBDatasetFile).filter(DBDatasetFile.dataset_id == id).delete()
    db_session.delete(dataset)
    db_session.commit()
    return dataset
