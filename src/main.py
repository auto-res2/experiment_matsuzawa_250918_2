import typer
import yaml
import torch
import numpy as np
import logging
import os
import timm

from . import preprocess
from . import train
from . import evaluate

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')

def load_config(config_path: str):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def set_seeds(seeds):
    seed = seeds[0] # Use the first seed for global setup
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_models(config):
    """Pre-loads model constructors to be used in evaluation."""
    models = {}
    all_model_names = set()
    for exp_params in config['evaluate'].values():
        for model_name in exp_params['models']:
            all_model_names.add(model_name)

    for model_name in all_model_names:
        # timm handles various model sources
        models[model_name] = lambda mn=model_name: timm.create_model(config['models'][mn]['source'], pretrained=True, num_classes=1000)
    
    return models


def main(smoke_test: bool = typer.Option(False, "--smoke-test", help="Run a quick smoke test."),
         full_experiment: bool = typer.Option(False, "--full-experiment", help="Run the full experiment.")):
    """
    Main entry point for the LANCE research experiment pipeline.
    """
    if not (smoke_test ^ full_experiment):
        logging.error("Please specify either --smoke-test or --full-experiment.")
        raise typer.Exit(code=1)

    config_path = 'config/smoke_test.yaml' if smoke_test else 'config/full_experiment.yaml'
    logging.info(f"Loading configuration from: {config_path}")
    config = load_config(config_path)
    
    # Update config for smoke test if applicable
    if smoke_test:
        smoke_params = config.get('smoke_test', {})
        config['preprocess']['batch_size'] = smoke_params.get('batch_size', 1)
        config['preprocess']['num_workers'] = 1
        config['train']['epochs'] = smoke_params.get('train_epochs', 1)
        config['train']['max_timesteps'] = smoke_params.get('train_timesteps', 100)
        for exp in config['evaluate']:
            config['evaluate'][exp]['max_frames'] = smoke_params.get('eval_frames', 100)

    # --- Setup ---
    set_seeds(config['global_settings']['seeds'])
    output_dir = config['global_settings']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Output directory set to: {output_dir}")

    # --- Phase 1: Data Preparation ---
    logging.info("PHASE 1: Preparing data streams...")
    dataloaders = preprocess.get_dataloaders(config)
    synthetic_generator = preprocess.create_synthetic_stream_generator(config)
    logging.info("Data preparation complete.")

    # --- Phase 2: Meta-Training ---
    logging.info("PHASE 2: Starting meta-training for LANCE...")
    # Use a template model for meta-training setup
    template_model_name = config['train']['template_model']
    template_model_source = config['models'][template_model_name]['source']
    template_model = timm.create_model(template_model_source, pretrained=True, num_classes=1000)
    model_artifacts_path = train.run_meta_training(config, template_model, synthetic_generator)
    logging.info("Meta-training complete.")

    # --- Phase 3: Evaluation ---
    logging.info("PHASE 3: Running evaluation experiments...")
    # Get constructors for all models needed in evaluation
    all_models = get_models(config)
    evaluate.run_experiments(config, dataloaders, model_artifacts_path, all_models)
    logging.info("Evaluation complete.")

    logging.info("Experiment pipeline finished successfully!")

if __name__ == "__main__":
    typer.run(main)
