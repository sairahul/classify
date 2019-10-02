
from fastai import *
from fastai.vision import *
from scipy.spatial import distance
import faiss
import numpy as np

from PIL import Image, ImageOps
import pydicom

from app.mlmodel.models import SiameseNet, get_learner


def is_dicom(fn):
    """True if the fn points to a DICOM image"""
    fn = str(fn)
    if fn.lower().endswith('.dcm'):
        return True

    # Dicom signature from the dicom spec.
    with open(fn,'rb') as fh:
        fh.seek(0x80)
        return fh.read(4) == b'DICM'


def open_image(fn:PathOrStr, div:bool=True, convert_mode:str='RGB', img_size:int=255):
    "Return `Image` object created from image in file `fn`."
    if is_dicom(fn):
        ds = pydicom.read_file(str(fn))
        if ds.PhotometricInterpretation.startswith('MONOCHROME'):
            max_size = ((1 << ds.BitsStored) - 1)
            im = np.stack([ds.pixel_array] * 3, -1)
            x = PIL.Image.fromarray(im)
        else:
            raise OSError('Unsupported DICOM image with PhotometricInterpretation=={}'.format(ds.PhotometricInterpretation))
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning) # EXIF warning from TiffPlugin
            x = PIL.Image.open(str(fn)).convert(convert_mode)
            max_size = 255

    x = PIL.ImageOps.fit(x, (img_size, img_size), PIL.Image.ANTIALIAS)
    x = pil2tensor(x, dtype=np.float32)
    if div: x.div_(max_size)
    return x


def create_feature_extraction_model(model, cut=None):
    body = create_body(model, cut=cut)
    for param in body.parameters():
        param.requires_grad = False
    return body


def extract_features(model, ids_list, files_list, batch_size=16):
    ids_list = np.int64(ids_list)
    image_features = []
    for i in range(0, len(files_list), batch_size):
        batch_files = files_list[i:i + batch_size]

        img_batch = []
        for img in batch_files:
            img = open_image(img)
            img_batch.append(img.data)

        img_batch = torch.stack(img_batch, 0)
        features = model(img_batch.cpu())
        features = features.cpu()
        print(features.shape)

        for img, feature in zip(batch_files, features):
            image_features.append(feature.detach().flatten().numpy())

    return ids_list, np.stack(image_features, axis=0)


def extract_features_siamese(model, ids_list, files_list, batch_size=512):
    ids_list = np.int64(ids_list)
    image_features = []
    for i in range(0, len(files_list), batch_size):
        batch_files = files_list[i:i + batch_size]

        img_batch = []
        for img in batch_files:
            img = open_image(img)
            img_batch.append(img.data)

        img_batch = torch.stack(img_batch, 0)
        features = model.get_embedding(img_batch.cuda())
        features = features.cpu()

        for img, feature in zip(batch_files, features):
            image_features.append(feature.flatten().numpy())

    return ids_list, np.stack(image_features, axis=0)


def get_siamese_net():
    snet = SiameseNet()
    snet.load_state_dict(torch.load("/data/models/siam_pnemothorax", map_location=torch.device('cpu')))
    snet.eval()
    return snet