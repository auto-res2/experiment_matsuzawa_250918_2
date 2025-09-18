import argparse
import yaml
import os
import sys
import time

# Ensure src is in python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from preprocess import get_data
from train import train_model
from evaluate import evaluate_model

def main():
    parser = argparse.ArgumentParser(description="Run APEX-Zero experiments.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--smoke-test', action='store_true', help='Run a small-scale smoke test.')
    group.add_argument('--full-experiment', action='store_true', help='Run the full-scale experiment.')

    args = parser.parse_args()

    if args.smoke_test:
        config_path = os.path.join(current_dir, '..', 'config', 'smoke_test.yaml')
    else:
        config_path = os.path.join(current_dir, '..', 'config', 'full_experiment.yaml')

    with open(config_path, 'r') as f:
        configs = yaml.safe_load(f)

    start_time = time.time()

    for exp_key, config in configs.items():
        if not isinstance(config, dict): continue # Skip metadata keys
        print(f"\n{'='*80}\nStarting Experiment: {config.get('name', exp_key)}\n{'='*80}")
        
        if args.smoke_test:
            config['smoke_test'] = True

        # 1. Load Data
        data = get_data(config)

        # 2. Train Model
        training_results = train_model(config, data)

        # 3. Evaluate Model
        evaluate_model(config, data, training_results)

    total_time = time.time() - start_time
    print(f"\n{'='*80}\nAll experiments completed in {total_time/60:.2f} minutes.\n{'='*80}")

if __name__ == '__main__':
    main()
