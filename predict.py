import os
import subprocess
import time

MODEL_CACHE = "models"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HOME"] = MODEL_CACHE
os.environ["TORCH_HOME"] = MODEL_CACHE
os.environ["HF_DATASETS_CACHE"] = MODEL_CACHE
os.environ["TRANSFORMERS_CACHE"] = MODEL_CACHE
os.environ["HUGGINGFACE_HUB_CACHE"] = MODEL_CACHE

import json
import numpy as np
import pandas as pd
from PIL import Image
from typing import List
import onnxruntime as rt
from pydantic import BaseModel
from cog import BasePredictor, Input, Path

from pydantic import BaseModel, Field


class Tag(BaseModel):
    tag: str
    confidence: float
    category: str


# Dataset v3 series of models:
SWINV2_MODEL_DSV3_REPO = "models/wd-swinv2-tagger-v3"
CONV_MODEL_DSV3_REPO = "models/wd-convnext-tagger-v3"
VIT_MODEL_DSV3_REPO = "models/wd-vit-tagger-v3"

# Dataset v2 series of models:
MOAT_MODEL_DSV2_REPO = "models/wd-v1-4-moat-tagger-v2"
SWIN_MODEL_DSV2_REPO = "models/wd-v1-4-swinv2-tagger-v2"
CONV_MODEL_DSV2_REPO = "models/wd-v1-4-convnext-tagger-v2"
CONV2_MODEL_DSV2_REPO = "models/wd-v1-4-convnextv2-tagger-v2"
VIT_MODEL_DSV2_REPO = "models/wd-v1-4-vit-tagger-v2"

# Files to load from the models directory
MODEL_FILENAME = "model.onnx"
LABEL_FILENAME = "selected_tags.csv"

kaomojis = [
    "0_0",
    "(o)_(o)",
    "+_+",
    "+_-",
    "._.",
    "<o>_<o>",
    "<|>_<|>",
    "=_=",
    ">_<",
    "3_3",
    "6_9",
    ">_o",
    "@_@",
    "^_^",
    "o_o",
    "u_u",
    "x_x",
    "|_|",
    "||_||",
]


def load_labels(dataframe) -> list[str]:
    name_series = dataframe["name"]
    name_series = name_series.map(
        lambda x: x.replace("_", " ") if x not in kaomojis else x
    )
    tag_names = name_series.tolist()

    rating_indexes = list(np.where(dataframe["category"] == 9)[0])
    general_indexes = list(np.where(dataframe["category"] == 0)[0])
    character_indexes = list(np.where(dataframe["category"] == 4)[0])
    return tag_names, rating_indexes, general_indexes, character_indexes


def mcut_threshold(probs):
    sorted_probs = probs[probs.argsort()[::-1]]
    difs = sorted_probs[:-1] - sorted_probs[1:]
    t = difs.argmax()
    thresh = (sorted_probs[t] + sorted_probs[t + 1]) / 2
    return thresh


def download_weights(url: str, dest: str) -> None:
    # NOTE WHEN YOU EXTRACT SPECIFY THE PARENT FOLDER
    start = time.time()
    print("[!] Initiating download from URL: ", url)
    print("[~] Destination path: ", dest)
    if ".tar" in dest:
        dest = os.path.dirname(dest)
    command = ["pget", "-vf" + ("x" if ".tar" in url else ""), url, dest]
    try:
        print(f"[~] Running command: {' '.join(command)}")
        subprocess.check_call(command, close_fds=False)
    except subprocess.CalledProcessError as e:
        print(
            f"[ERROR] Failed to download weights. Command '{' '.join(e.cmd)}' returned non-zero exit status {e.returncode}."
        )
        raise
    print("[+] Download completed in: ", time.time() - start, "seconds")


class Predictor(BasePredictor):
    def setup(self):
        model_files = [
            "wd-convnext-tagger-v3.tar",
            "wd-swinv2-tagger-v3.tar",
            "wd-v1-4-convnext-tagger-v2.tar",
            "wd-v1-4-convnextv2-tagger-v2.tar",
            "wd-v1-4-moat-tagger-v2.tar",
            "wd-v1-4-swinv2-tagger-v2.tar",
            "wd-v1-4-vit-tagger-v2.tar",
            "wd-vit-tagger-v3.tar",
        ]

        base_url = (
            f"https://weights.replicate.delivery/default/wd-tagger/{MODEL_CACHE}/"
        )

        if not os.path.exists(MODEL_CACHE):
            os.makedirs(MODEL_CACHE)

        for model_file in model_files:
            url = base_url + model_file
            filename = url.split("/")[-1]
            dest_path = os.path.join(MODEL_CACHE, filename)
            if not os.path.exists(dest_path.replace(".tar", "")):
                download_weights(url, dest_path)
            else:
                print(f"[~] Skipping download, file already exists: {filename}")

        self.model_target_size = None
        self.last_loaded_repo = None
        self.load_model(SWINV2_MODEL_DSV3_REPO)  # Load default model

    def load_model(self, model_repo):
        if model_repo == self.last_loaded_repo:
            return

        csv_path = os.path.join(model_repo, LABEL_FILENAME)
        model_path = os.path.join(model_repo, MODEL_FILENAME)

        tags_df = pd.read_csv(csv_path)
        sep_tags = load_labels(tags_df)

        self.tag_names = sep_tags[0]
        self.rating_indexes = sep_tags[1]
        self.general_indexes = sep_tags[2]
        self.character_indexes = sep_tags[3]

        model = rt.InferenceSession(model_path)
        _, height, width, _ = model.get_inputs()[0].shape
        self.model_target_size = height

        self.last_loaded_repo = model_repo
        self.model = model

    def prepare_image(self, image):
        target_size = self.model_target_size

        # Create a canvas with the same size as the image
        canvas = Image.new("RGBA", image.size, (255, 255, 255))
        image = image.convert("RGBA")  # Ensure image is in RGBA mode
        canvas.alpha_composite(image)
        image = canvas.convert("RGB")

        image_shape = image.size
        max_dim = max(image_shape)
        pad_left = (max_dim - image_shape[0]) // 2
        pad_top = (max_dim - image_shape[1]) // 2

        padded_image = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
        padded_image.paste(image, (pad_left, pad_top))

        if max_dim != target_size:
            padded_image = padded_image.resize(
                (target_size, target_size),
                Image.BICUBIC,
            )

        image_array = np.asarray(padded_image, dtype=np.float32)
        image_array = image_array[:, :, ::-1]

        return np.expand_dims(image_array, axis=0)

    def predict(
        self,
        image: Path = Input(
            description="Path to the input image file to be analyzed by the WaifuDiffusion tagger model"
        ),
        model_repo: str = Input(
            description="Name of the pre-trained model repository to use for image analysis",
            default=SWINV2_MODEL_DSV3_REPO,
            choices=[
                SWINV2_MODEL_DSV3_REPO,
                CONV_MODEL_DSV3_REPO,
                VIT_MODEL_DSV3_REPO,
                MOAT_MODEL_DSV2_REPO,
                SWIN_MODEL_DSV2_REPO,
                CONV_MODEL_DSV2_REPO,
                CONV2_MODEL_DSV2_REPO,
                VIT_MODEL_DSV2_REPO,
            ],
        ),
        general_thresh: float = Input(
            description="Probability threshold for including general tags in the output (between 0 and 1)",
            default=0.35,
            ge=0,
            le=1,
        ),
        general_mcut_enabled: bool = Input(
            description="Whether to use the MCut algorithm to automatically determine the general tags threshold",
            default=False,
        ),
        character_thresh: float = Input(
            description="Probability threshold for including character tags in the output (between 0 and 1)",
            default=0.85,
            ge=0,
            le=1,
        ),
        character_mcut_enabled: bool = Input(
            description="Whether to use the MCut algorithm to automatically determine the character tags threshold",
            default=False,
        ),
        category: str = Input(
            description="Category of tags to return in the output",
            default="all_tags",
            choices=["all_tags", "general", "character", "rating"],
        ),
    ) -> List[Tag]:
        image = Image.open(str(image))
        self.load_model(model_repo)

        image = self.prepare_image(image)

        input_name = self.model.get_inputs()[0].name
        label_name = self.model.get_outputs()[0].name
        preds = self.model.run([label_name], {input_name: image})[0]

        labels = list(zip(self.tag_names, preds[0].astype(float)))

        ratings_names = [labels[i] for i in self.rating_indexes]
        rating = dict(ratings_names)

        general_names = [labels[i] for i in self.general_indexes]

        if general_mcut_enabled:
            general_probs = np.array([x[1] for x in general_names])
            general_thresh = mcut_threshold(general_probs)

        general_res = [x for x in general_names if x[1] > general_thresh]
        general_res = dict(general_res)

        character_names = [labels[i] for i in self.character_indexes]

        if character_mcut_enabled:
            character_probs = np.array([x[1] for x in character_names])
            character_thresh = mcut_threshold(character_probs)
            character_thresh = max(0.15, character_thresh)

        character_res = [x for x in character_names if x[1] > character_thresh]
        character_res = dict(character_res)

        tags = []
        if category == "all_tags" or category == "general":
            for tag, score in general_res.items():
                tags.append(Tag(tag=tag, confidence=score, category="general"))
        if category == "all_tags" or category == "character":
            for tag, score in character_res.items():
                tags.append(Tag(tag=tag, confidence=score, category="character"))
        if category == "all_tags" or category == "rating":
            for tag, score in rating.items():
                tags.append(Tag(tag=tag, confidence=score, category="rating"))

        return tags
