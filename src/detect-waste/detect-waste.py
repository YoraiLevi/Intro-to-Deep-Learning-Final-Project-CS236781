"""
AI4Good project for detecting waste in environment. www.detectwaste.ml.

Our latest results were published in Waste Management journal in article titled Deep learning-based waste detection in natural and urban environments.

You can find more technical details in our technical report Waste detection in Pomerania: non-profit project for detecting waste in environment.

Did you know that we produce 300 million tons of plastic every year? And only the part of it is properly recycled.

The idea of detect waste project is to use Artificial Intelligence to detect plastic waste in the environment. Our solution is applicable for video and photography. Our goal is to use AI for Good.

In Detect Waste in Pomerania project we used 9 publicity available datasets, and additional data collected using Google Images Download.

For more details, about the data we used, check our jupyter notebooks with data exploratory analysis.
"""

from pathlib import Path
import csv
import datasets
import itertools
import json
import os
import PIL.Image

# BibTeX citation
_CITATION = """\
@article{MAJCHROWSKA2022274,
title = {Deep learning-based waste detection in natural and urban environments},
journal = {Waste Management},
volume = {138},
pages = {274-284},
year = {2022},
issn = {0956-053X},
doi = {https://doi.org/10.1016/j.wasman.2021.12.001},
url = {https://www.sciencedirect.com/science/article/pii/S0956053X21006474},
author = {Sylwia Majchrowska and Agnieszka Mikołajczyk and Maria Ferlin and Zuzanna Klawikowska and Marta A. Plantykow and Arkadiusz Kwasigroch and Karol Majek},
keywords = {Object detection, Semi-supervised learning, Waste classification benchmarks, Waste detection benchmarks, Waste localization, Waste recognition},
abstract = {Waste pollution is one of the most significant environmental issues in the modern world. The importance of recycling is well known, both for economic and ecological reasons, and the industry demands high efficiency. Current studies towards automatic waste detection are hardly comparable due to the lack of benchmarks and widely accepted standards regarding the used metrics and data. Those problems are addressed in this article by providing a critical analysis of over ten existing waste datasets and a brief but constructive review of the existing Deep Learning-based waste detection approaches. This article collects and summarizes previous studies and provides the results of authors’ experiments on the presented datasets, all intended to create a first replicable baseline for litter detection. Moreover, new benchmark datasets detect-waste and classify-waste are proposed that are merged collections from the above-mentioned open-source datasets with unified annotations covering all possible waste categories: bio, glass, metal and plastic, non-recyclable, other, paper, and unknown. Finally, a two-stage detector for litter localization and classification is presented. EfficientDet-D2 is used to localize litter, and EfficientNet-B2 to classify the detected waste into seven categories. The classifier is trained in a semi-supervised fashion making the use of unlabeled images. The proposed approach achieves up to 70% of average precision in waste detection and around 75% of classification accuracy on the test dataset. The code and annotations used in the studies are publicly available online11https://github.com/wimlds-trojmiasto/detect-waste..}
}
"""

_DESCRIPTION = """\
AI4Good project for detecting waste in environment. www.detectwaste.ml.

Our latest results were published in Waste Management journal in article titled Deep learning-based waste detection in natural and urban environments.

You can find more technical details in our technical report Waste detection in Pomerania: non-profit project for detecting waste in environment.

Did you know that we produce 300 million tons of plastic every year? And only the part of it is properly recycled.

The idea of detect waste project is to use Artificial Intelligence to detect plastic waste in the environment. Our solution is applicable for video and photography. Our goal is to use AI for Good.
"""

_HOMEPAGE = "https://github.com/wimlds-trojmiasto/detect-waste"

_LICENSE = (
    "https://raw.githubusercontent.com/wimlds-trojmiasto/detect-waste/main/LICENSE"
)

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


def exif_transpose(image, *, in_place=False, remove_exif_metadata_only=False):
    from PIL import ExifTags, Image

    """
    https://pillow.readthedocs.io/en/stable/_modules/PIL/ImageOps.html#exif_transpose
    If an image has an EXIF Orientation tag, other than 1, transpose the image
    accordingly, and remove the orientation data.

    :param image: The image to transpose.
    :param in_place: Boolean. Keyword-only argument.
        If ``True``, the original image is modified in-place, and ``None`` is returned.
        If ``False`` (default), a new :py:class:`~PIL.Image.Image` object is returned
        with the transposition applied. If there is no transposition, a copy of the
        image will be returned.
    """
    image_exif = image.getexif()
    orientation = image_exif.get(ExifTags.Base.Orientation)

    method = {
        2: Image.Transpose.FLIP_LEFT_RIGHT,
        3: Image.Transpose.ROTATE_180,
        4: Image.Transpose.FLIP_TOP_BOTTOM,
        5: Image.Transpose.TRANSPOSE,
        6: Image.Transpose.ROTATE_270,
        7: Image.Transpose.TRANSVERSE,
        8: Image.Transpose.ROTATE_90,
    }.get(orientation)
    if method is not None:
        transposed_image = image
        if not remove_exif_metadata_only:
            transposed_image = image.transpose(method)
        if in_place:
            image.im = transposed_image.im
            image.pyaccess = None
            image._size = transposed_image._size
        exif_image = image if in_place else transposed_image

        exif = exif_image.getexif()
        if ExifTags.Base.Orientation in exif:
            del exif[ExifTags.Base.Orientation]
            if "exif" in exif_image.info:
                exif_image.info["exif"] = exif.tobytes()
            elif "Raw profile type exif" in exif_image.info:
                exif_image.info["Raw profile type exif"] = exif.tobytes().hex()
            elif "XML:com.adobe.xmp" in exif_image.info:
                for pattern in (
                    r'tiff:Orientation="([0-9])"',
                    r"<tiff:Orientation>([0-9])</tiff:Orientation>",
                ):
                    exif_image.info["XML:com.adobe.xmp"] = re.sub(
                        pattern, "", exif_image.info["XML:com.adobe.xmp"]
                    )
        if not in_place:
            return transposed_image
    elif not in_place:
        return image.copy()


class DetectWasteDataset(datasets.GeneratorBasedBuilder):
    """AI4Good project for detecting waste in environment. www.detectwaste.ml."""

    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="taco-multi",
            version=VERSION,
            description="TACO dataset with 7 detect-waste categories object detection training",
        ),
    ]
    DEFAULT_CONFIG_NAME = "taco-multi"

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
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    # fmt: off
    # manually verified.
    # these images need to be rotated so fit the annotations
    _image_ids_for_exif_transpose = [228,229,230,231,232,233,234,235,237,240,241,242,243,295,296,297,298,299,300,301,302,303,304,307,308,309,310,311,312,313,315,318,319,320,321,322,323,324,325,326,327,328,329,330,331,332,333,334,335,336,337,338,339,340,341,342,343,345,353,354,355,359,360,362,372,460,522,527,528,529,530,531,532,534,535,536,537,538,539,540,541,542,543,545,546,548,550,551,553,556,557,559,560,561,562,563,564,565,566,567,568,569,570,571,572,573,574,575,576,577,578,579,580,581,582,583,584,585,586,587,588,589,590,591,592,593,594,595,596,597,598,599,600,601,602,603,604,605,606,607,608,609,610,611,612,613,614,615,617,618,672,679,680]
    # these images do not need to be rotated, but have exif metadata that needs to be removed to cleanliness
    _image_ids_for_exif_metadata_removal = [1501,1509,1511,1516,1518,1520,1530,1535,1538,1539,1541,1542,1545,1549,1550,1552,1559,1562,1563,1564,1568,1574,1576,1579,1581,1582,1584,1585,1590,1591,1592,1600,3274,3275,3294,3297,3298,3303,3304,3312,3313,3316,3317,3323,3326,3327,3331,3339,3340,3347,3350,3352,3354,3356,3364,3365,3372,3373,3376,3377,3378,3380,3382,3386,3390,3395,3397,3398,3399,3400,3401,3403,3406,3411,3412,3413,3415,3418,3421,3422,3427,3431,3432,3435,3437,3440,3442,3444,3447,3450,3452,3453,3461,3463,3464,3465,3466,3467,3474,3476,3478,3479,3480,3481,3492,3497,3499,3500,3501,3505,3508,3515,3516,3518,3522,3528,3529,3530,3533,3534,3536,3540,3541,3543,3548,3556,3558,3559,3565,3569,3570,3576,3579,3583,3584,3591,3592,3595,3599,3601,3606,3607,3609,3610,3613,3617,3623,3624,3625,3626,3629,3630,3633,3644,3645,3650,3655,3660,3663,3664,3665,3666,3668,3673,3678,3679,3680,3683,3686,3688,3690,3691,3693,3694,3695,3697,3701,3702,3704,3710,3711,3712,3717,3720,3723,3726,3730,3731,3732,3741,3742,3743,3745,3747,3750,3757,3758,3759,3763,3764,3767,3769,3773,3776,3780,3783,3786,3789,3790,3795,3796,3798,3800,3804,3805,3813,3815,3818,3820,3822,3823,3824,3825,3831,3833,3836,3837,3840,3841,3843,3852,3854,3855,3859,3863,3866,3869,3875,3878,3880,3884,3885,3886,3890,3891,3892,3896,3898,3900,3906,3908,3914,3916,3920,3923,3927,3929,3931,3932,3936,3937,3938,3939,3944,3947,3948,3952,3953,3958,3959,3960,3966,3967,3969,3971,3973,3979,3980,3981,3982,3987,3992,3994,3996,3997,4000,4001,4002,4004,4009,4010,4013,4014,4016,4017,4020,4021,4022,4028,4031,4034,4035,4037,4039,4043,4044,4047,4049,4050,4058,4061,4063,4066,4069,4072,4076,4079,4080,4085,4090,4094,4096,4098,4100,4101,4103,4104,4105,4107,4109,4111,4114,4116,4117,4118,4119,4122,4123,4124,4130,4134,4137,4139,4141,4143,4152,4155,4156,4158,4162,4164,4171,4173,4174,4175,4180,4181,4186,4188,4189,4191,4193,4200,4203,4204,4206,4207,4213,4214,4221,4224,4227,4229,4230,4233,4236,4245,4246,4248,4249,4250,4254,4255,4258,4259,4262,4263,4265,4272,4274,4282,4286,4289,4290,4296,4297,4299,4304,4306,4310,4313,4316,4321,4322,4323,4326,4331,4335,4336,4339,4348,4350,4360,4365,4367,4368,4369,4373,4378,4380,4390,4392,4396,4397,4399,4409,4411,4412,4413,4414,4416,4420,4423,4426,4427,4434,4439,4441,4446,4453,4454,4460,4462,4471,4472,4473,4475,4476,4479,4481,4483,4494,4497,4499,4504,4506,4508,4510,4513,4516,4518,4519,4520,4521,4522,4524,4527,4528,4533,4534,4537,4541,4545,4547,4550,4558,4563,4566,4569,4572,4577,4580,4581,4583,4586,4588,4591,4592,4593,4597,4599,4602,4603,4606,4609,4613,1507,1523,1536,1537,1555,1595,3279,3283,3284,3328,3358,3371,3388,3394,3408,3423,3425,3428,3433,3434,3449,3490,3504,3519,3523,3564,3567,3578,3589,3622,3640,3654,3658,3659,3667,3681,3687,3689,3698,3715,3718,3725,3727,3737,3749,3753,3756,3761,3768,3799,3835,3838,3844,3873,3881,3883,3907,3912,3935,3946,3957,3964,3974,3983,4033,4038,4051,4064,4075,4084,4091,4093,4120,4138,4159,4160,4166,4183,4192,4219,4220,4231,4240,4269,4291,4308,4311,4314,4317,4341,4342,4345,4358,4364,4372,4375,4393,4400,4431,4442,4445,4448,4449,4464,4480,4525,4540,4551,4555,4560,4561,4568,4585,4590,4605,4607]
    # fmt: on
    def _split_generators(self, dl_manager):
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

        images_train = annotations_train["images"]
        images_test = annotations_test["images"]
        # sanity check - train,test don't have overlapping images
        assert not set(map(lambda image: image["id"], images_train)) & set(
            map(lambda image: image["id"], images_test)
        )

        all_images = itertools.chain(images_train, images_test)
        all_image_annotations = itertools.chain(
            annotations_train["annotations"], annotations_test["annotations"]
        )

        image_id2image = {image["id"]: image for image in all_images}
        image_ids_for_download = list(
            map(
                lambda ann: ann["image_id"],
                all_image_annotations,
            )
        )

        def standardize_image_dict(image):
            name = Path(image["file_name"]).name
            image["flickr_640_url"] = image.get("flickr_640_url") or ""
            image["flickr_url"] = image.get("flickr_url") or ""
            if name in images:
                image.update(images[name])

        def get_download_url(image):
            file_path = Path(image["file_name"])
            url = (
                image["flickr_640_url"]
                if Path(image["flickr_640_url"]).name == file_path.name
                else image["flickr_url"]
            )
            return url

        files = {}
        for image_id in image_ids_for_download:
            image = image_id2image[image_id]
            standardize_image_dict(image)
            url = get_download_url(image)
            files[image_id] = url

        image_id2storage_path = dl_manager.download(files)

        def standardize_image_data(image, remove_exif_metadata_only=False):
            image_id = image["id"]
            image_name = Path(image["file_name"])
            image_storage_path = Path(image_id2storage_path[image_id])
            format = image_name.suffix[1:].lower().replace("jpg", "jpeg")

            new_image_storage_path = image_storage_path.with_stem(
                image_name.stem + "_exif"
            )
            with PIL.Image.open(str(image_storage_path)) as I:
                exif_transpose(
                    I,
                    in_place=True,
                    remove_exif_metadata_only=remove_exif_metadata_only,
                )
                I.save(
                    str(new_image_storage_path),
                    format=format,
                    subsampling=0,
                    quality=100,
                )
            image_id2storage_path[image_id] = str(new_image_storage_path.absolute())

        for image_id in self._image_ids_for_exif_transpose:
            image = image_id2image[image_id]
            standardize_image_data(image)
        for image_id in self._image_ids_for_exif_metadata_removal:
            image = image_id2image[image_id]
            standardize_image_data(image, remove_exif_metadata_only=True)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "image_id2storage_path": image_id2storage_path,
                    "annotations": annotations_train,
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "image_id2storage_path": image_id2storage_path,
                    "annotations": annotations_test,
                    "split": "test",
                },
            ),
        ]

    def _generate_examples(self, image_id2storage_path, annotations, split):
        all_image_annotations = annotations["annotations"]
        all_image_annotations = sorted(
            all_image_annotations, key=lambda ann: ann["image_id"]
        )
        image_id2grouped_annotations = {
            image_id: list(group)
            for image_id, group in itertools.groupby(
                all_image_annotations, lambda x: x["image_id"]
            )
        }

        all_images = annotations["images"]
        image_id2image = {image["id"]: image for image in all_images}

        for image_id in image_id2image:
            file_path = image_id2storage_path[image_id]
            width, height = (
                image_id2image[image_id]["width"],
                image_id2image[image_id]["height"],
            )
            objects = image_id2grouped_annotations[image_id]
            objects = [
                {
                    "id": object["id"],
                    "area": object["area"],
                    "bbox": object["bbox"],
                    "category": object["category_id"] - 1,
                }
                for object in objects
            ]
            dat = image_id, {
                "image_id": image_id,
                "image": file_path,
                "width": width,
                "height": height,
                "objects": objects,
            }
            yield dat
