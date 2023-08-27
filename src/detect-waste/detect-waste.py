# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# TODO: Address all TODOs and remove all explanatory comments
"""TODO: Add a description here."""

# %%
import csv
import json
import os
from pathlib import Path
import itertools
import PIL.Image
from PIL import ImageOps
import datasets


# TODO: Add BibTeX citation
# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """\
@InProceedings{huggingface:dataset,
title = {A great new dataset},
author={huggingface, Inc.
},
year={2020}
}
"""

# TODO: Add description of the dataset here
# You can copy an official description
_DESCRIPTION = """\
This new dataset is designed to solve this great NLP task and is crafted with a lot of care.
"""

# TODO: Add a link to an official homepage for the dataset here
_HOMEPAGE = ""

# TODO: Add the licence for the dataset here if you can find it
_LICENSE = ""

# TODO: Add link to the official dataset URLs here
# The HuggingFace Datasets library doesn't host the datasets but only points to the original files.
# This can be an arbitrary nested dict/list of URLs (see below in `_split_generators` method)
_URLS = {
    "annotations_train": "https://raw.githubusercontent.com/wimlds-trojmiasto/detect-waste/main/annotations/annotations_train.json",
    "annotations_test": "https://raw.githubusercontent.com/wimlds-trojmiasto/detect-waste/main/annotations/annotations_test.json",
    "taco_all_image_urls_csv": "https://raw.githubusercontent.com/pedropro/TACO/master/data/all_image_urls.csv",
}

_CATEGORIES = [
    "metals_and_plastic",
    "other",
    "non_recyclable",
    "glass",
    "paper",
    "bio",
    "unknown",
]


# TODO: Name of the dataset usually matches the script name with CamelCase instead of snake_case
class TacoDetectWasteDataset(datasets.GeneratorBasedBuilder):
    """TODO: Short description of my dataset."""

    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="multi",
            version=VERSION,
            description="TACO dataset with 7 detect-waste categories object detection training",
        ),
        # datasets.BuilderConfig(
        #     name="binary",
        #     version=VERSION,
        #     description="TACO dataset with 1 class object detection training",
        # ),
        datasets.BuilderConfig(
            name="mini-multi",
            version=VERSION,
            description="A subset of multi with 10 images",
        ),
        datasets.BuilderConfig(
            name="localfiles",
            version=VERSION,
            description="A subset of multi with 10 images",
        ),
        datasets.BuilderConfig(
            name="mini-localfiles",
            version=VERSION,
            description="A subset of multi with 10 images",
        ),
    ]

    DEFAULT_CONFIG_NAME = "multi"  # It's not mandatory to have a default configuration. Just use one if it make sense.

    def _info(self):
        features = datasets.Features(
            {
                "image_id": datasets.Value("int64"),
                "image": datasets.Image(),
                "width": datasets.Value("int32"),
                "height": datasets.Value("int32"),
                "objects": datasets.Sequence(
                    { 
                        "id": datasets.Value("int64"),
                        "area": datasets.Value("int64"),
                        "bbox": datasets.Sequence(datasets.Value("float32"), length=4),
                        "category": datasets.ClassLabel(names=_CATEGORIES),
                    }
                ),
            }
        )
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=features,  # Here we define them above because they are different between the two configurations
            # If there's a common (input, target) tuple from the features, uncomment supervised_keys line below and
            # specify them. They'll be used if as_supervised=True in builder.as_dataset.
            # supervised_keys=("sentence", "label"),
            # Homepage of the dataset for documentation
            homepage=_HOMEPAGE,
            # License for the dataset if available
            license=_LICENSE,
            # Citation for the dataset
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager, local_dir = None):
        # TODO: This method is tasked with downloading/extracting the data and defining the splits depending on the configuration
        # If several configurations are possible (listed in BUILDER_CONFIGS), the configuration selected by the user is in self.config.name

        # dl_manager is a datasets.download.DownloadManager that can be used to download and extract URLS
        # It can accept any type or nested list/dict and will give back the same structure with the url replaced with path to local files.
        # By default the archives will be extracted and a path to a cached folder where they are extracted is returned instead of the archive
        # urls = _URLS[self.config.name]
        # data_dir = dl_manager.download_and_extract(urls)
        data_dir = dl_manager.download(_URLS)
        csv_path = data_dir["taco_all_image_urls_csv"]
        train_json_path = data_dir["annotations_train"]
        test_json_path = data_dir["annotations_test"]

        annotations_train = json.load(open(train_json_path))
        annotations_test = json.load(open(test_json_path))

        images = {}
        with open(csv_path) as f:
            reader = csv.DictReader(f, fieldnames=["flickr_640_url", "flickr_url"])
            for row in reader:
                for url in row.values():
                    if url:
                        images[Path(url).name] = row

        def standardize_image_dict(image):
            name = Path(image["file_name"]).name
            image["flickr_640_url"] = image.get("flickr_640_url") or ""
            image["flickr_url"] = image.get("flickr_url") or ""
            if name in images:
                image.update(images[name])

        for image in annotations_train["images"]:
            standardize_image_dict(image)
        for image in annotations_test["images"]:
            standardize_image_dict(image)

        ids_to_download_train = set(
            map(lambda ann: ann["image_id"], annotations_train["annotations"])
        )
        images_to_download_train = list(
            filter(
                lambda image: image["id"] in ids_to_download_train,
                annotations_train["images"],
            )
        )

        ids_to_download_test = set(
            map(lambda ann: ann["image_id"], annotations_test["annotations"])
        )
        images_to_download_test = list(
            filter(
                lambda image: image["id"] in ids_to_download_test,
                annotations_test["images"],
            )
        )

        def download_url(image):
            file_path = Path(image["file_name"])
            url = (
                image["flickr_640_url"]
                if Path(image["flickr_640_url"]).name == file_path.name
                else image["flickr_url"]
            )
            return image["id"], url
            
        urls_train = dict(
            map(lambda image: download_url(image), images_to_download_train)
        )
        urls_test = dict(
            map(lambda image: download_url(image), images_to_download_test)
        )
        if self.config.name == "mini-multi":
            urls_train = dict(itertools.islice(urls_train.items(), 10))
            urls_test = dict(itertools.islice(urls_test.items(), 10))
        files_train = None
        files_test = None
       
        if "localfiles" in self.config.name:
            base_dir = Path("./TACO/data")
            files_train = {
               image["id"]: str(base_dir / image["file_name"]) for image in images_to_download_train
            }
            files_test = {
               image["id"]: str(base_dir / image["file_name"]) for image in images_to_download_test
            }
        else:
            files_train = dl_manager.download(urls_train)
            files_test = dl_manager.download(urls_test)
        from itertools import chain
        # from concurrent import futures
        # with futures.ThreadPoolExecutor() as executor:
        # for image_path in chain(files_train.values(),files_test.values()):
            # ImageOps.exif_transpose(PIL.Image.open(image_path)).save(image_path, 'JPEG', subsampling=0, quality=100)
                # executor.submit(self.exif_image,(image_path))
                    
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "files": files_train,
                    "annotations": annotations_train,
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "files": files_test,
                    "annotations": annotations_test,
                    "split": "test",
                },
            ),
        ]

    # def exif_image(self, image_path):
    #     with PIL.Image.open(image_path) as I:
    #         # print(image_path, I.width,I.height)
    #         ImageOps.exif_transpose(I,in_place=True)
    #         # print(image_path, I.width,I.height)
    #         I.save(image_path)
    
    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, files, annotations, split):
        
        anns_anns = filter(lambda ann: ann["image_id"] in files, annotations["annotations"])
        anns_anns = sorted(anns_anns, key=lambda ann: ann["image_id"])
        anns_anns = {
            key: list(group)
            for key, group in itertools.groupby(anns_anns, lambda x: x["image_id"])
        }
        anns_image = filter(lambda ann: ann["id"] in files, annotations["images"])
        anns_image = {ann["id"]: ann for ann in anns_image}
        i = 0
        
        
        for image_id, file_path in files.items():
            if('mini' in self.config.name):
                if(i >= 10):
                    break
            width, height = anns_image[image_id]['width'],anns_image[image_id]['height'] 
            # I = PIL.Image.open(file_path)
            # if I.width != width or I.height != height:
                # I = exif_transpose(I)
            #     print(image_id,"I")
            #     print(f"{(I.width != width or I.height != height)=}")
            #     I = ImageOps.exif_transpose(I,in_place=True)
            #     print(f"{(I.width != width or I.height != height)=}")
            objects = anns_anns[image_id]
            objects = [{
                'id' : object['id'],
                'area': object['area'],
                'bbox': object['bbox'],
                'category': object['category_id'] - 1,
                } for object in objects]
            dat =  image_id, {
                "image_id": image_id,
                "image": file_path,
                "width" : width,
                "height": height,
                "objects": objects
            }
            i +=1
            yield dat
# %%
