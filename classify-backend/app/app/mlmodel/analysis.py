
from typing import List
import os

from fastai.vision.models import resnet34
import faiss

from app.db_models.dataset import Dataset, DatasetFile, AnalysisModel
from app.mlmodel.utils import create_feature_extraction_model, extract_features, get_siamese_net

ANALYSIS_MODELS = {
    "resnet34": {
        "name": "Resnet34",
        "model": create_feature_extraction_model(resnet34),
        "size": 512*8*8,
        "img_size": 128
    },
    "siamese": {
        "name": "Siamese",
        "model": get_siamese_net(),
        "size": 128,
        "img_size": 128
    }
}


def _get_files(dataset_files: List[DatasetFile], base_dir: str):
    files = []
    ids = []
    for dataset_file in dataset_files:
        fn = os.path.join(base_dir, dataset_file.path)
        files.append(fn)
        ids.append(dataset_file.id)
    return ids, files


def create_index(dataset: Dataset, model_name: str, base_dir: str, model_id: int):
    """
    https://github.com/sairahul/notebooks/blob/master/resnet-feature-extraction.ipynb
    """
    model = ANALYSIS_MODELS[model_name]

    ids, files = _get_files(dataset.files, dataset.base_dir)

    output_dir = os.path.join(base_dir, f"{dataset.id}")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"index_{model_id}")

    ids, extracted_features = extract_features(model["model"], ids, files)
    index = faiss.IndexFlatL2(model["size"])
    index = faiss.IndexIDMap(index)
    index.add_with_ids(extracted_features, ids)
    faiss.write_index(index, output_file)
    return output_file


def find_similar_images(dataset_file: DatasetFile, analysis_model: AnalysisModel):
    model = ANALYSIS_MODELS[analysis_model.name]

    #dataset_id = dataset_file.dataset_id
    #fn = os.path.join(base_dir, f"{dataset_id}/index")
    fn = analysis_model.index

    ids, files = _get_files([dataset_file], dataset_file.dataset.base_dir)
    ids, extracted_features = extract_features(model["model"], ids, files)

    index = faiss.read_index(fn)
    similar_ids = index.search(extracted_features, 32)
    return similar_ids

