"""PyTorch dataset adapter for SPE3R multi-spacecraft pretraining."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np

from utils.postprocess import quat2dcm

import json
import logging
from pathlib import Path

import pandas as pd
import torch


logger = logging.getLogger(__name__)


class SPE3RDataset(torch.utils.data.Dataset):
    """Index SPE3R samples using its official spacecraft and image splits.

    Official SPE3R split:

    Spacecraft level
    ----------------
    - 57 spacecraft assigned to train
    - 7 spacecraft assigned to test

    Image level for each training spacecraft
    ----------------------------------------
    - Train:
      img000001-img000400
      img000501-img000900

    - Validation:
      img000401-img000500
      img000901-img001000
    """

    def __init__(
        self,
        cfg,
        split="train",
        transforms=None,
        target_generators=None,
    ):
        if split not in {"train", "val", "test"}:
            raise ValueError(
                "split must be 'train', 'val', or 'test'"
            )

        # For SPE3R, ROOT points directly to the folder containing:
        #
        # camera.json
        # splits.csv
        # spacecraft folders
        self.root = Path(cfg.DATASET.ROOT)

        self.split = split
        self.is_train = split == "train"
        self.is_supervised = split in {"train", "val"}

        self.image_size = tuple(cfg.DATASET.IMAGE_SIZE)
        self.input_size = tuple(cfg.DATASET.INPUT_SIZE)

        self.output_size = tuple(
            int(size / cfg.DATASET.OUTPUT_SIZE[0])
            for size in self.input_size
        )

        self.head_names = (
            list(cfg.MODEL.HEAD.LOSS_HEADS)
            if self.is_supervised
            else list(cfg.TEST.HEAD)
        )

        # SPE3R spacecraft do not share Tango's 11 keypoints.
        if "heatmap" in self.head_names:
            raise ValueError(
                "The heatmap head cannot be used during SPE3R "
                "pretraining because SPE3R does not provide "
                "shared Tango-style keypoints."
            )

        self.load_masks = "segmentation" in self.head_names
        self.transforms = transforms
        self.target_generators = target_generators
        self.load_labels = target_generators is not None

        # Optional subset used for:
        # - the current one-spacecraft audit;
        # - pilot experiments;
        # - spacecraft-diversity ablations.
        #
        # An empty list means: use the complete official split.
        self.requested_spacecraft = list(
            getattr(cfg.DATASET, "SPACECRAFT", [])
        )

        self.samples, self.spacecraft_names = (
            self._build_sample_index()
        )

        logger.info(
            "SPE3R %s split: %d spacecraft, %d images",
            self.split,
            len(self.spacecraft_names),
            len(self.samples),
        )

    def __len__(self):
        """Return the number of indexed images."""

        return len(self.samples)

    
    def __getitem__(self, index):
        """Load one SPE3R image and its training targets."""

        if not 0 <= index < len(self):
            raise IndexError(
                f"Sample index {index} is outside "
                f"the dataset range 0-{len(self) - 1}."
            )

        sample = self.samples[index]

        # SPE3R is converted to grayscale to match the
        # official SPNv2 preprocessing of SPEED+.
        image = self._load_image(sample)

        # Some ODR-style operations may request images
        # without any labels.
        if not self.load_labels:
            if self.transforms is not None:
                image = self.transforms(image=image)["image"]

            return image

        # The full-resolution mask is always needed because
        # SPE3R does not provide bounding boxes separately.
        mask = self._load_mask(sample)

        # Bounding box is initially represented using
        # normalized coordinates in [0, 1].
        boundingbox = self._bounding_box_from_mask(mask)

        transform_kwargs = {
            "image": image,
            "bboxes": [boundingbox],
            "class_labels": [sample["spacecraft"]],
        }

        if self.transforms is not None:
            transformed = self.transforms(**transform_kwargs)

            image = transformed["image"]

            boundingbox = np.asarray(
                transformed["bboxes"][0],
                dtype=np.float32,
            )

        # Convert normalized bounding-box coordinates
        # into input-image pixels, as expected by SPNv2.
        boundingbox *= np.asarray(
            [
                self.input_size[0],
                self.input_size[1],
                self.input_size[0],
                self.input_size[1],
            ],
            dtype=np.float32,
        )

        quaternion = np.asarray(
            sample["quaternion"],
            dtype=np.float32,
        )

        quaternion_norm = np.linalg.norm(quaternion)

        if quaternion_norm <= 1e-8:
            raise ValueError(
                f"Invalid zero quaternion for "
                f"{sample['spacecraft']}/"
                f"{sample['filename']}."
            )

        # Normalization does not change the represented
        # orientation; it only protects against numerical
        # rounding errors.
        quaternion = quaternion / quaternion_norm

        translation = np.asarray(
            sample["translation"],
            dtype=np.float32,
        )

        rotationmatrix = quat2dcm(
            quaternion
        ).astype(np.float32)

        targets = {
            "domain": sample["spacecraft"],
            "boundingbox": torch.from_numpy(
                boundingbox
            ),
            "quaternion": torch.from_numpy(
                quaternion
            ),
            "rotationmatrix": torch.from_numpy(
                rotationmatrix
            ),
            "translation": torch.from_numpy(
                translation
            ),
        }

        if self.is_supervised and self.load_masks:
            mask_output = cv2.resize(
                mask,
                self.output_size,
                interpolation=cv2.INTER_NEAREST,
            )

            mask_output = (
                mask_output > 128
            ).astype(np.float32)

            targets["mask"] = torch.from_numpy(
                mask_output
            ).unsqueeze(0)

        return image, targets

    def _load_image(self, sample):
        """Load one SPE3R image as three-channel grayscale."""

        image_path = (
            self.root
            / sample["spacecraft"]
            / "images"
            / f"{sample['filename']}.jpg"
        )

        image_gray = cv2.imread(
            str(image_path),
            cv2.IMREAD_GRAYSCALE,
        )

        if image_gray is None:
            raise FileNotFoundError(
                f"Unable to read SPE3R image: "
                f"{image_path}"
            )

        expected_shape = (
            self.input_size[1],
            self.input_size[0],
        )

        if image_gray.shape != expected_shape:
            raise ValueError(
                f"{image_path} has shape "
                f"{image_gray.shape}; expected "
                f"{expected_shape}. SPE3R images should "
                f"remain at native resolution."
            )

        # Produce three identical channels so the image
        # remains compatible with the ImageNet-initialized
        # EfficientNet backbone.
        image = cv2.cvtColor(
            image_gray,
            cv2.COLOR_GRAY2BGR,
        )

        return image

    def _load_mask(self, sample):
        """Load SPE3R's binary spacecraft mask."""

        mask_path = (
            self.root
            / sample["spacecraft"]
            / "masks"
            / f"{sample['filename']}.png"
        )

        mask_raw = cv2.imread(
            str(mask_path),
            cv2.IMREAD_UNCHANGED,
        )

        if mask_raw is None:
            raise FileNotFoundError(
                f"Unable to read SPE3R mask: "
                f"{mask_path}"
            )

        if mask_raw.ndim == 2:
            mask = mask_raw

        elif (
            mask_raw.ndim == 3
            and mask_raw.shape[2] >= 1
        ):
            # SPE3R stores the same binary mask in the
            # first three channels. The alpha channel is
            # fully opaque and must not be used.
            mask = mask_raw[:, :, 0]

        else:
            raise ValueError(
                f"Unexpected SPE3R mask shape "
                f"{mask_raw.shape} at {mask_path}."
            )

        expected_shape = (
            self.input_size[1],
            self.input_size[0],
        )

        if mask.shape != expected_shape:
            raise ValueError(
                f"{mask_path} has shape {mask.shape}; "
                f"expected {expected_shape}."
            )

        # Guarantee a clean binary mask containing
        # only background 0 and spacecraft 255.
        mask = np.where(
            mask > 128,
            255,
            0,
        ).astype(np.uint8)

        if not np.any(mask):
            raise ValueError(
                f"Empty spacecraft mask: {mask_path}"
            )

        return mask

    @staticmethod
    def _bounding_box_from_mask(mask):
        """Calculate normalized [xmin, ymin, xmax, ymax]."""

        rows, columns = np.nonzero(mask)

        if columns.size == 0:
            raise ValueError(
                "Cannot calculate a bounding box "
                "from an empty mask."
            )

        image_height, image_width = mask.shape

        # This follows the same convention used by the
        # official SPNv2 SPEED+ preprocessing script.
        boundingbox = np.asarray(
            [
                columns.min() / image_width,
                rows.min() / image_height,
                columns.max() / image_width,
                rows.max() / image_height,
            ],
            dtype=np.float32,
        )

        return boundingbox
    

    def _build_sample_index(self):
        """Create the list of allowed SPE3R samples."""

        split_path = self.root / "splits.csv"

        if not split_path.is_file():
            raise FileNotFoundError(
                f"SPE3R split file not found: {split_path}"
            )

        # splits.csv has no header.
        split_table = pd.read_csv(
            split_path,
            header=None,
            names=["spacecraft", "split"],
        )

        # Train and validation use the 57 training spacecraft.
        # Test uses only the seven official held-out spacecraft.
        spacecraft_split = (
            "test" if self.split == "test" else "train"
        )

        official_names = split_table.loc[
            split_table["split"] == spacecraft_split,
            "spacecraft",
        ].astype(str).tolist()

        if self.requested_spacecraft:
            invalid_names = sorted(
                set(self.requested_spacecraft)
                - set(official_names)
            )

            if invalid_names:
                raise ValueError(
                    "The following spacecraft do not belong to "
                    f"the official {spacecraft_split} split: "
                    f"{invalid_names}"
                )

            selected_names = list(self.requested_spacecraft)

        else:
            # Empty subset means the complete official split.
            selected_names = official_names

        samples = []

        for spacecraft in selected_names:
            spacecraft_root = self.root / spacecraft

            labels_path = spacecraft_root / "labels.json"
            images_dir = spacecraft_root / "images"
            masks_dir = spacecraft_root / "masks"

            required_paths = [
                labels_path,
                images_dir,
                masks_dir,
            ]

            missing_paths = [
                path
                for path in required_paths
                if not path.exists()
            ]

            if missing_paths:
                missing_text = ", ".join(
                    str(path) for path in missing_paths
                )

                raise FileNotFoundError(
                    f"Incomplete SPE3R spacecraft "
                    f"'{spacecraft}'. Missing: {missing_text}"
                )

            with labels_path.open(
                "r",
                encoding="utf-8",
            ) as file:
                labels = json.load(file)

            spacecraft_samples = []

            for label in labels:
                filename = str(label["filename"])

                image_number = self._parse_image_number(
                    filename
                )

                if not self._image_belongs_to_split(
                    image_number
                ):
                    continue

                spacecraft_samples.append(
                    {
                        "spacecraft": spacecraft,
                        "filename": filename,
                        "quaternion":
                            label["q_vbs2tango_true"],
                        "translation":
                            label["r_Vo2To_vbs_true"],
                    }
                )

            expected_count = (
                1000
                if self.split == "test"
                else 800
                if self.split == "train"
                else 200
            )

            if len(spacecraft_samples) != expected_count:
                raise ValueError(
                    f"{spacecraft} produced "
                    f"{len(spacecraft_samples)} "
                    f"{self.split} samples; "
                    f"expected {expected_count}."
                )

            spacecraft_samples.sort(
                key=lambda sample: sample["filename"]
            )

            samples.extend(spacecraft_samples)

        if not samples:
            raise ValueError(
                f"No SPE3R samples were found "
                f"for split '{self.split}'."
            )

        return samples, selected_names

    @staticmethod
    def _parse_image_number(filename):
        """Convert img000001 into the integer 1."""

        stem = Path(filename).stem

        if (
            not stem.startswith("img")
            or not stem[3:].isdigit()
        ):
            raise ValueError(
                f"Unexpected SPE3R filename: '{filename}'"
            )

        return int(stem[3:])

    def _image_belongs_to_split(self, image_number):
        """Apply the official SPE3R image-level split."""

        if self.split == "train":
            return (
                1 <= image_number <= 400
                or 501 <= image_number <= 900
            )

        if self.split == "val":
            return (
                401 <= image_number <= 500
                or 901 <= image_number <= 1000
            )

        # All images of the seven held-out spacecraft
        # belong to the optional test split.
        return 1 <= image_number <= 1000