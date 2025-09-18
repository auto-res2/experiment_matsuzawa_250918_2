import torch
import torchvision.transforms as transforms
from datasets import load_dataset, concatenate_datasets
from torch.utils.data import DataLoader, IterableDataset
import logging
import itertools
from typing import Optional, List

class StreamDataset(IterableDataset):
    """Simple iterable dataset that yields (image_tensor, label) tuples."""

    def __init__(self, hf_dataset, transform, max_samples=None):
        super().__init__()
        self.dataset = hf_dataset
        self.transform = transform
        self.max_samples = max_samples

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        iterable = iter(self.dataset)

        # Optional truncation (used in smoke-tests)
        if self.max_samples is not None:
            iterable = itertools.islice(iterable, self.max_samples)

        # Distribute data between workers (very simple striding strategy)
        if worker_info is not None and worker_info.num_workers > 1:
            iterable = itertools.islice(iterable, worker_info.id, None, worker_info.num_workers)

        for sample in iterable:
            image = sample["image"]
            if image.mode != "RGB":
                image = image.convert("RGB")

            # Many datasets use different label field names – try common ones
            label = (
                sample.get("label")
                if "label" in sample
                else sample.get("fine_label", -1)
            )

            if self.transform:
                image = self.transform(image)
            yield image, label


def get_transform(config, task="classification"):
    if task == "classification":
        return transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=config["preprocess"]["mean"],
                    std=config["preprocess"]["std"],
                ),
            ]
        )
    elif task == "segmentation":
        return transforms.Compose(
            [
                transforms.Resize((1024, 2048)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=config["preprocess"]["cityscapes_mean"],
                    std=config["preprocess"]["cityscapes_std"],
                ),
            ]
        )
    else:
        raise ValueError(f"Unknown task for transform: {task}")


# -----------------------------------------------------------------------------
# Helper for robust dataset download / selection
# -----------------------------------------------------------------------------

def _select_split(ds_dict, preferred: Optional[str] = None):
    """Return a single datasets.Dataset from a DatasetDict.

    If `preferred` is supplied and present in the dict, that split is returned. Otherwise the
    first available split (sorted by name) is returned. This prevents the
    'concatenate_datasets' error when accidentally passing DatasetDicts.
    """

    if not isinstance(ds_dict, dict):
        # Already a `datasets.Dataset`
        return ds_dict

    splits = list(ds_dict.keys())
    if preferred and preferred in ds_dict:
        return ds_dict[preferred]
    # fall back to first split
    return ds_dict[splits[0]]


def download_and_prepare_dataset(
    name: str,
    subset: Optional[str],
    split: Optional[str],
    cache_dir: str,
):
    """Download a HuggingFace dataset and return **a single split Dataset**.

    This wrapper enforces the fail-fast behaviour required by the project and makes sure we
    always return a `datasets.Dataset` (never a `DatasetDict`) so that later concatenation
    works without additional checks.
    """

    try:
        logging.info(
            f"Loading dataset '{name}' | subset={subset} | split={split or 'auto'}"
        )
        ds = load_dataset(name, subset, split=split, cache_dir=cache_dir)
        # `load_dataset` returns a Dataset when `split` is specified, otherwise a
        # DatasetDict – guard against the latter just in case.
        ds = _select_split(ds, split)
        return ds
    except Exception as e:
        logging.error(f"Failed to load dataset {name} ({subset=}): {e}")
        raise RuntimeError(
            "Dataset unavailable – experiment aborted as per NO-FALLBACK constraint."
        )


# -----------------------------------------------------------------------------
# Main public API used by the pipeline
# -----------------------------------------------------------------------------

def get_dataloaders(config):
    logging.info("Preparing dataloaders…")

    cache_dir = config["preprocess"]["cache_dir"]
    batch_size = config["preprocess"]["batch_size"]
    num_workers = config["preprocess"]["num_workers"]

    cls_transform = get_transform(config, "classification")
    seg_transform = get_transform(config, "segmentation")

    dataloaders = {}

    for stream_name, stream_cfg in config["preprocess"]["streams"].items():
        logging.info(f"Creating stream '{stream_name}'")
        hf_datasets: List = []

        for ds_info in stream_cfg["datasets"]:
            # Default split selection
            split = ds_info.get("split", "validation")

            if ds_info["name"] == "imagenet-c":
                # Expand corruption × severity grid
                for corruption in ds_info["corruptions"]:
                    for severity in ds_info["severities"]:
                        subset_name = f"{corruption}_{severity}"
                        d = download_and_prepare_dataset(
                            "hendrycks/imagenet-c",
                            subset=subset_name,
                            split="test",  # ImageNet-C only provides a 'test' split
                            cache_dir=cache_dir,
                        )
                        hf_datasets.append(d)
            else:
                d = download_and_prepare_dataset(
                    ds_info["name"],
                    subset=ds_info.get("subset"),
                    split=split,
                    cache_dir=cache_dir,
                )
                hf_datasets.append(d)

        if not hf_datasets:
            logging.warning(f"No datasets found for stream '{stream_name}', skipping.")
            continue

        # Concatenate all datasets belonging to this stream (they are all individual
        # `datasets.Dataset` objects at this point).
        combined_dataset = concatenate_datasets(hf_datasets).shuffle(
            seed=config["global_settings"]["seeds"][0]
        )

        transform = seg_transform if stream_cfg["task"] == "segmentation" else cls_transform
        max_samples = config.get("smoke_test", {}).get("max_frames_per_stream")

        stream_dataset = StreamDataset(combined_dataset, transform, max_samples)
        dataloaders[stream_name] = DataLoader(
            stream_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
        )

    logging.info("Dataloaders prepared.")
    return dataloaders


def create_synthetic_stream_generator(config):
    """Create an endless synthetic data stream used during meta-training."""

    logging.info("Creating synthetic stream generator for meta-training.")
    params = config["train"]["synthetic_stream"]
    cache_dir = config["preprocess"]["cache_dir"]

    # A lightweight public dataset is sufficient – e.g. CIFAR-10 for smoke-test, ImageNet-1k for full run
    source_dataset = load_dataset(
        params["source_dataset"],
        split=params.get("split", "train"),
        cache_dir=cache_dir,
    ).shuffle(
        seed=config["global_settings"]["seeds"][0]
    )

    synthetic_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.5, contrast=0.5, saturation=0.5, hue=0.3
            ),
            transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 5.0)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=config["preprocess"]["mean"],
                std=config["preprocess"]["std"],
            ),
        ]
    )

    iterable_ds = StreamDataset(source_dataset, synthetic_transform)
    return DataLoader(
        iterable_ds,
        batch_size=params["batch_size"],
        num_workers=config["preprocess"]["num_workers"],
    )
