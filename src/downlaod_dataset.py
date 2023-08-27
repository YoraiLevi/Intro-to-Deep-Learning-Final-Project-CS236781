#%%

import json
import os.path
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed, wait
from io import BytesIO
from pathlib import Path

import pandas as pd
import requests
from PIL import Image
from tqdm import tqdm

csv_path = (
    r".\TACO\data\all_image_urls.csv"
)

train_path = r".\detect-waste\annotations\annotations_train.json"
test_path = r".\detect-waste\annotations\annotations_test.json"


def standardize_image_dict(image):
    name = Path(image["file_name"]).name
    image["flickr_640_url"] = image.get("flickr_640_url") or ""
    image["flickr_url"] = image.get("flickr_url") or ""
    if name in images:
        image.update(images[name])


def get_url(image):
    file_path = Path(image["file_name"])
    url = (
        image["flickr_640_url"]
        if Path(image["flickr_640_url"]).name == file_path.name
        else image["flickr_url"]
    )
    return str(file_path), url


def download_url(url, path: Path):
    if not path.is_file():
        response = requests.get(url)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            f.write(response.content)

df = pd.read_csv(csv_path, header=None, names=["flickr_640_url", "flickr_url"])
df["flickr_name"] = df["flickr_url"].map(lambda url: Path(str(url)).name)
df["flickr_640_name"] = df["flickr_640_url"].map(lambda url: Path(str(url)).name)
images = {}
for i, row in df.iterrows():
    d = {
        "flickr_url": row["flickr_url"] if not pd.isna(row["flickr_url"]) else "",
        "flickr_640_url": row["flickr_640_url"]
        if not pd.isna(row["flickr_640_url"])
        else "",
    }
    images[row["flickr_name"]] = d
    images[row["flickr_640_name"]] = d
annotations_json_paths = [test_path, train_path]
base = Path("./TACO/data/")
if __name__ == "__main__":
    name_urls = {}
    for annotations_json_path in annotations_json_paths:
        with open(annotations_json_path, "r") as f:
            annotations = json.loads(f.read())
        for image in annotations["images"]:
            standardize_image_dict(image)
        name_urls.update(dict(map(get_url, annotations["images"])))
    with ThreadPoolExecutor() as exec:
        try:
            with tqdm(total=len(name_urls)) as pbar:
                futures_list = [
                    exec.submit(download_url, url, base / file_name)
                    for file_name, url in name_urls.items()
                ]
                # wait(futures_list)
                for future in as_completed(futures_list):
                    pbar.update(1)
        except KeyboardInterrupt:
            exec.shutdown(wait=True, cancel_futures=True)
#%%


# %%
import PIL.Image
from PIL import ImageOps
Iexif = ImageOps.exif_transpose(I)

# %%
def has_exif(I):
    exif = I.getexif()
    orientation = exif.get(0x0112, 1)  # default 1
    if orientation != 1:
        return True
    return False
def filter_image(image, I = None):
    if not I:
        I = PIL.Image.open(base / image['file_name'])
    return (I.width != image['width'] or I.height != image['height'])
bad_images = list(filter(filter_image,annotations["images"]))
for image in bad_images:
    I = PIL.Image.open(base / image['file_name'])
    file_name = Path(image['file_name'])
    exif_fixed_filename = file_name.with_stem(file_name.stem+"_exif")
    Iexif = ImageOps.exif_transpose(I)
    Iexif.save(exif_fixed_filename.name)
#%%
for image in bad_images:
    file_name = Path(image['file_name'])
    exif_fixed_filename = file_name.with_stem(file_name.stem+"_exif")
    I = PIL.Image.open(exif_fixed_filename.name)
    print(exif_fixed_filename.name, filter_image(image, I))
    

# %%
image = bad_images[0]
file_name = Path(image['file_name'])
exif_fixed_filename = file_name.with_stem(file_name.stem+"_exif")
I = PIL.Image.open(exif_fixed_filename.name)
print(exif_fixed_filename.name, filter_image(image, I))
# %%
import json
import PIL.Image
from PIL import ImageOps
from pathlib import Path 
base = Path("./TACO/data/")
train_path = r".\detect-waste\annotations\annotations_train.json"
test_path = r".\detect-waste\annotations\annotations_test.json"
annotations_json_path = test_path 
with open(annotations_json_path, "r") as f:
            annotations = json.loads(f.read())
dict_images = {image['id']: image for image in annotations['images']}
def has_exif(I):
    exif = I.getexif()
    orientation = exif.get(0x0112, 1)  # default 1
    if orientation != 1:
        return True
    return False
def filter_image(image, I = None):
    if not I:
        I = PIL.Image.open(base / image['file_name'])
    return (I.width != image['width'] or I.height != image['height'])
bad_images = list(filter(filter_image,annotations["images"]))
for image in bad_images:
    file_name = Path(image['file_name'])
    file_path = base / file_name
    with PIL.Image.open(file_path) as I:
        Iexif = ImageOps.exif_transpose(I)
        Iexif.save(file_path)
for image in bad_images:
    file_name = Path(image['file_name'])
    file_path = base / file_name
    with PIL.Image.open(file_path) as I:
        print(file_path, filter_image(image, I))