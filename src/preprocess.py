import os
import torch
from torch_geometric.datasets import Planetoid, Reddit, Flickr
from ogb.nodeproppred import PygNodePropPredDataset


def get_data(config):
    dataset_name = config['dataset']['name']
    data_dir = os.path.join('data', dataset_name)
    os.makedirs(data_dir, exist_ok=True)

    print(f"\nLoading dataset: {dataset_name}")

    try:
        if dataset_name.lower() == 'reddit':
            dataset = Reddit(root=data_dir)
        elif dataset_name.lower() == 'flickr':
            dataset = Flickr(root=data_dir)
        elif dataset_name.startswith('ogbn-'):
            dataset = PygNodePropPredDataset(name=dataset_name, root=data_dir)
        elif dataset_name.lower() in ['cora', 'citeseer', 'pubmed']:
             dataset = Planetoid(root=data_dir, name=dataset_name)
        else:
            raise RuntimeError(f"Dataset '{dataset_name}' not supported. Aborting experiment as per NO-FALLBACK policy.")
    except Exception as e:
        print(f"Error loading dataset {dataset_name}: {e}")
        raise RuntimeError(f"Dataset '{dataset_name}' could not be loaded or downloaded. Aborting experiment as per NO-FALLBACK policy.")

    data = dataset[0]

    # OGB datasets need split indices
    if dataset_name.startswith('ogbn-'):
        split_idx = dataset.get_idx_split()
        data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.train_mask[split_idx['train']] = True
        data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.val_mask[split_idx['valid']] = True
        data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.test_mask[split_idx['test']] = True
        data.y = data.y.squeeze() # OGB labels are often [N, 1]

    # For smoke tests, create a smaller subset
    if config.get('smoke_test', False):
        print("Creating a smaller subset for smoke test.")
        num_nodes = data.num_nodes
        subset_nodes = torch.randperm(num_nodes)[:int(num_nodes * 0.1)] # 10% of nodes
        data = data.subgraph(subset_nodes)

    print(f"Dataset '{dataset_name}' loaded successfully:")
    print(data)
    return data
