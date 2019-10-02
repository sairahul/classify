from typing import List
import os
import os.path
from pathlib import Path
import json

from fastapi import APIRouter, Depends, HTTPException

from app import crud
from app.api.utils.db import get_db
from app.db_models.dataset import Dataset as DBDataset, DatasetFile as DBDatasetFile, AnalysisModel as DBAnalysisModel
from app.models.dataset import Dataset, DatasetCreate, DatasetFile, DatasetFiles, AnalysisModelIn, AnalysisModel, DatasetModel
from app.db.session import Session
from app.core.config import DATASET_STATIC_ORIG_TEMPLATE, DATASET_STATIC_THUMB_TEMPLATE, ANALYSIS_INDEX_DIR
from app.mlmodel.analysis import create_index, find_similar_images

router = APIRouter()


def get_image_path(dataset_id, rel_path):
    image = DATASET_STATIC_ORIG_TEMPLATE.format(dataset_id=dataset_id)
    return os.path.join(image, rel_path)


def get_thumb_path(dataset_id, rel_path):
    image = DATASET_STATIC_THUMB_TEMPLATE.format(dataset_id=dataset_id)
    return os.path.join(image, rel_path)


@router.get("/list", response_model=List[Dataset])
def read_items(
    db: Session = Depends(get_db),
    skip: int = 0,
    limit: int = 100
):
    """
    Retrieve items.
    """
    items = db.query(DBDataset).offset(skip).limit(limit).all()
    return items


@router.post("/", response_model=Dataset)
def create_item(
    *,
    db: Session = Depends(get_db),
    dataset_in: DatasetCreate,
):
    """
    Create new item.
    """
    base_dir = Path(dataset_in.base_dir)
    thumbnail_dir = Path(dataset_in.thumbnail_dir)
    if (not base_dir.exists()) or (not thumbnail_dir.exists()):
        raise HTTPException(status_code=400, detail=f"Invalid path {base_dir} or {thumbnail_dir}")

    dataset = crud.dataset.create(db_session=db, dataset_in=dataset_in)
    files = []
    for each_file in base_dir.glob("**/*"):
        if each_file.is_dir():
            continue
        rel_path = os.path.relpath(each_file, base_dir)
        thumbnail = thumbnail_dir.joinpath(rel_path)

        if thumbnail.exists():
            _thumbnail = thumbnail
        elif thumbnail.with_suffix(".jpg").exists():
            _thumbnail = thumbnail.with_suffix(".jpg")
        else:
            crud.dataset.remove(db_session=db, id=dataset.id)
            raise HTTPException(status_code=400, detail=f"Thumbnail {thumbnail} doesn't exist")

        thumbnail_rel_path = os.path.relpath(_thumbnail, thumbnail_dir)
        dataset_file = DBDatasetFile(name=each_file.name, dataset_id=dataset.id,
                                     path=rel_path, thumbnail=thumbnail_rel_path)
        files.append(dataset_file)

    db.bulk_save_objects(files)
    db.commit()
    return dataset


@router.delete("/{id}", response_model=Dataset)
def delete_item(
    *,
    db: Session = Depends(get_db),
    id: int
):
    """
    Delete an item.
    """
    item = crud.dataset.get(db_session=db, id=id)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    item = crud.dataset.remove(db_session=db, id=id)
    return item


def _get_dataset_model_out(dataset_files: List[DBDatasetFile], dataset_id: int):
    items = []
    for dataset_file in dataset_files:
        meta = dataset_file.meta if dataset_file.meta else "{}"
        items.append(
            DatasetFile(id=dataset_file.id, image=get_image_path(dataset_id, dataset_file.path),
                        thumbnail=get_thumb_path(dataset_id, dataset_file.thumbnail),
                        meta=json.loads(meta))
        )
    return items


@router.get('/browse/{dataset_id}', response_model=DatasetFiles)
def dataset_browse( *,
    db: Session = Depends(get_db),
    dataset_id: int,
    skip: int = 0,
    limit: int = 100
):
    dataset = crud.dataset.get(db_session=db, id=dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail=f"dataset {dataset_id} not found")

    analysis_models = db.query(DBAnalysisModel).filter(DBAnalysisModel.dataset_id==dataset_id).all()
    result = {
        'datasets': [dataset]
    }

    dataset_files = (db.query(DBDatasetFile)
                        .filter(DBDatasetFile.dataset_id == dataset_id).offset(skip)
                        .limit(limit)
                        .all())

    result['items'] = _get_dataset_model_out(dataset_files, dataset_id)
    result['models'] = [DatasetModel(id=m.id, name=m.name) for m in analysis_models]
    return result


@router.post('/create_index/', response_model=AnalysisModel)
def dataset_index( *,
    db: Session = Depends(get_db),
    analysis_model_in: AnalysisModelIn
):
    dataset = crud.dataset.get(db_session=db, id=analysis_model_in.dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail=f"dataset {analysis_model_in.dataset_id} not found")

    analysis_model = DBAnalysisModel(index="", name=analysis_model_in.name,
                                     dataset_id=analysis_model_in.dataset_id)
    db.add(analysis_model)
    db.commit()
    db.refresh(analysis_model)

    index_location = create_index(dataset, analysis_model_in.name, ANALYSIS_INDEX_DIR, analysis_model.id)
    analysis_model.index = index_location
    db.commit()

    return AnalysisModel(id=analysis_model.id, index=analysis_model.index, name=analysis_model.name,
                         dataset_id=analysis_model.dataset_id)


@router.get('/similar/{model_id}/{image_id}', response_model=DatasetFiles)
def dataset_index(*, db: Session = Depends(get_db), model_id: int, image_id: int):

    dataset_file = db.query(DBDatasetFile).filter(DBDatasetFile.id == image_id).first()
    model_file = db.query(DBAnalysisModel).filter(DBAnalysisModel.id == model_id).first()
    
    distance, similar_ids = find_similar_images(dataset_file, model_file)
    similar_ids = [int(x) for x in similar_ids[0] if int(x) != image_id]
    similar_dataset_files = db.query(DBDatasetFile).filter(DBDatasetFile.id.in_(similar_ids)).all()
    similar_ids_map = {dataset_file.id: dataset_file for dataset_file in similar_dataset_files}
    similar_ids_order = [similar_ids_map[each_id] for each_id in similar_ids]

    result = {
        'datasets': [dataset_file.dataset],
        'items': _get_dataset_model_out(similar_ids_order, dataset_file.dataset.id),
        'models': []
    }
    return result


@router.post('/update_metadata/', response_model=str)
def update_metadata(*,
    db: Session = Depends(get_db),
    dataset_id: int,
):
    import pydicom
    import csv

    '''
    dataset = crud.dataset.get(db_session=db, id=dataset_id)
    for each_file in dataset.files:
        fn = os.path.join(dataset.base_dir, each_file.path)
    if not dataset:
        raise HTTPException(status_code=404, detail=f"dataset {dataset_id} not found")

    for each_file in dataset.files:
    '''

    '''
    labels = {}
    with open("/data/datasets/siim-acr-pneumothorax-segmentation/labels.csv") as fp:
        reader = csv.reader(fp)
        headers = next(reader)
        for _id, label in reader:
            labels[_id] = label.strip()

    attrs = ["PatientID", "PatientName", "PatientAge", "PatientSex", "ViewPosition", "BodyPartExamined",
             "Modality", "StudyDate", "StudyTime"]

    notfound = 0
    for each_file in dataset.files:
        fn = os.path.join(dataset.base_dir, each_file.path)
        ds = pydicom.dcmread(fn)
        _id = each_file.path.split("/")[-1].rsplit(".", 1)[0]
        if _id in labels:
            meta = {"HasPneumothorax": labels[_id]}
        else:
            notfound += 1
            meta = {}
        meta.update({attr: str(getattr(ds, attr, '')) for attr in attrs})
        each_file.meta = json.dumps(meta)

    print(f"Not found: {notfound}")
    '''
    db.commit()
    return "OK"

