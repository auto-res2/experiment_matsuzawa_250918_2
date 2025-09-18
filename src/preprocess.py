import torch
import torchvision.transforms as transforms
from datasets import load_dataset, concatenate_datasets
from torch.utils.data import DataLoader, IterableDataset
import logging
import itertools

class StreamDataset(IterableDataset):
    def __init__(self, hf_dataset, transform, max_samples=None):
        super().__init__()
        self.dataset = hf_dataset
        self.transform = transform
        self.max_samples = max_samples

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        iterable = iter(self.dataset)
        if self.max_samples:
            iterable = itertools.islice(iterable, self.max_samples)

        if worker_info is not None:
            # If in a worker process, split the workload.
            # This is a simple split, may not be perfect for all datasets.
            iterable = itertools.islice(iterable, worker_info.id, None, worker_info.num_workers)

        for sample in iterable:
            image = sample['image']
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            label_key = 'label' if 'label' in sample else 'fine_label' # For Cityscapes
            label = sample.get(label_key, -1) 
            
            if self.transform:
                image = self.transform(image)
            yield image, label

def get_transform(config, task='classification'):
    if task == 'classification':
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=config['preprocess']['mean'], std=config['preprocess']['std'])
        ])
    elif task == 'segmentation':
        return transforms.Compose([
            transforms.Resize((1024, 2048)),
            transforms.ToTensor(),
            transforms.Normalize(mean=config['preprocess']['cityscapes_mean'], std=config['preprocess']['cityscapes_std'])
        ])
    else:
        raise ValueError(f"Unknown task for transform: {task}")


def download_and_prepare_dataset(name, config, split, cache_dir):
    try:
        logging.info(f"Attempting to load dataset: {name} with split: {split}")
        if config:
            return load_dataset(name, config, split=split, cache_dir=cache_dir)
        else:
            return load_dataset(name, split=split, cache_dir=cache_dir)
    except Exception as e:
        logging.error(f"Failed to load dataset {name}: {e}")
        raise RuntimeError('Dataset unavailable – experiment aborted as per NO-FALLBACK constraint.')


def get_dataloaders(config):
    logging.info("Preparing dataloaders...")
    dataloaders = {}
    cache_dir = config['preprocess']['cache_dir']
    batch_size = config['preprocess']['batch_size']
    num_workers = config['preprocess']['num_workers']

    cls_transform = get_transform(config, 'classification')
    seg_transform = get_transform(config, 'segmentation')

    for stream_name, stream_config in config['preprocess']['streams'].items():
        logging.info(f"Creating stream: {stream_name}")
        hf_datasets = []
        for ds_info in stream_config['datasets']:
            split = ds_info.get('split', 'validation')
            # ImageNet-C has many subsets
            if ds_info['name'] == 'imagenet-c': 
                for corruption in ds_info['corruptions']:
                    for severity in ds_info['severities']:
                        subset_name = f'{corruption}_{severity}'
                        d = download_and_prepare_dataset('haideraltahan/wds_imagenetc', subset_name, split=None, cache_dir=cache_dir)
                        hf_datasets.append(d)
            else:
                d = download_and_prepare_dataset(ds_info['name'], config, split=split, cache_dir=cache_dir)
                hf_datasets.append(d)

        # Concatenate and shuffle
        if not hf_datasets:
            logging.warning(f"No datasets found for stream {stream_name}, skipping.")
            continue

        combined_dataset = concatenate_datasets(hf_datasets).shuffle(seed=config['global_settings']['seeds'][0])
        
        transform = seg_transform if stream_config['task'] == 'segmentation' else cls_transform
        max_samples = config.get('smoke_test', {}).get('max_frames_per_stream', None)

        stream_dataset = StreamDataset(combined_dataset, transform, max_samples=max_samples)
        
        dataloaders[stream_name] = DataLoader(
            stream_dataset, 
            batch_size=batch_size, 
            num_workers=num_workers
        )

    logging.info("Dataloaders prepared.")
    return dataloaders

def create_synthetic_stream_generator(config):
    logging.info("Creating synthetic stream generator for meta-training.")
    params = config['train']['synthetic_stream']
    cache_dir = config['preprocess']['cache_dir']
    
    # Use a small, readily available dataset as a source
    source_dataset = load_dataset(params['source_dataset'], split='train', cache_dir=cache_dir).shuffle()
    
    # Strong, randomized augmentations
    synthetic_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.3),
        transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 5.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config['preprocess']['mean'], std=config['preprocess']['std'])
    ])
    
    # This will be an endless stream due to IterableDataset nature
    iterable_ds = StreamDataset(source_dataset, synthetic_transform)

    return DataLoader(
        iterable_ds,
        batch_size=params['batch_size'],
        num_workers=config['preprocess']['num_workers']
    )
