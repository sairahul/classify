import requests
import argparse
import os
from pathlib import Path

from PIL import Image, ImageOps
import pydicom

URL = "http://localhost:8000/api/v1/datasets/"


def is_dicom(fn):
    """True if the fn points to a DICOM image"""
    fn = str(fn)
    if fn.lower().endswith('.dcm'):
        return True

    # Dicom signature from the dicom spec.
    with open(fn,'rb') as fh:
        fh.seek(0x80)
        return fh.read(4) == b'DICM'


def main(args):
    base_dir = Path(args.base_dir)
    thumbnail_dir = Path(args.thumbnail_dir)

    if (not base_dir.exists()) or (not thumbnail_dir.exists()):
        print(f"{base_dir} not found.")
        return

    for each_file in base_dir.glob("**/*"):
        if each_file.is_dir():
            continue

        rel_path = os.path.relpath(each_file, base_dir)
        dir_name = os.path.dirname(rel_path)
        file_name = os.path.basename(rel_path)

        _thumbnail_dir = thumbnail_dir.joinpath(dir_name)
        _thumbnail = _thumbnail_dir.joinpath(file_name)
        os.makedirs(_thumbnail_dir, exist_ok=True)

        if is_dicom(each_file):
            ds = pydicom.read_file(str(each_file))
            image = Image.fromarray(ds.pixel_array)
            _thumbnail = _thumbnail.with_suffix(".jpg")
        else:
            image = Image.open(each_file)

        thumb = ImageOps.fit(image, (128, 128), Image.ANTIALIAS)
        thumb.save(_thumbnail)

    post_data = {
        "name": args.name,
        "description": args.desc,
        "base_dir": args.base_dir,
        "thumbnail_dir": args.thumbnail_dir
    }
    dataset = requests.post(URL, json=post_data)
    print(f"Response: {dataset.status_code}")
    print(f"Response: {dataset.text}")


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Add a new dataset")
    parser.add_argument("--name", help="Dataset name")
    parser.add_argument("--desc", default="", help="Dataset description")
    parser.add_argument("--base-dir", help="Base directory. All files in this directory get indexed")
    parser.add_argument("--thumbnail-dir", help="Thumbnail ")
    args = parser.parse_args()
    main(args)

