import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis
from sklearn.metrics import roc_auc_score
import time
import json
import os
import logging
from tqdm import tqdm
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
from pynvml.smi import nvidia_smi
from codecarbon import OfflineEmissionsTracker
import fvcore.nn

from .train import LowRankMod, HyperRNN

# SECTION: ADAPTERS

class BaseAdapter:
    def __init__(self, model, config, model_artifacts_path=None):
        self.model = model
        self.config = config
        self.device = next(model.parameters()).device
        self.reset()

    def step(self, batch):
        raise NotImplementedError

    def reset(self):
        pass

class SourceOnlyAdapter(BaseAdapter):
    def step(self, batch):
        images, _ = batch
        images = images.to(self.device)
        return self.model(images)

class BNAdaptAdapter(BaseAdapter):
    def __init__(self, model, config, model_artifacts_path=None):
        super().__init__(model, config, model_artifacts_path)
        self.model.train() # Set to train mode to update BN stats

    def step(self, batch):
        images, _ = batch
        images = images.to(self.device)
        return self.model(images)

    def reset(self):
        # Reset BN stats if needed (not standard for TTA)
        pass

def get_tent_parameters(model):
    params = []
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
            if m.weight is not None:
                params.append(m.weight)
            if m.bias is not None:
                params.append(m.bias)
    return params

class TentAdapter(BaseAdapter):
    def __init__(self, model, config, model_artifacts_path=None):
        super().__init__(model, config, model_artifacts_path)
        self.steps = config.get('steps', 5)
        self.optimizer = torch.optim.Adam(get_tent_parameters(model), lr=config.get('lr', 1e-3))
        self.model.train() # Enable gradient computation and BN updates

    def step(self, batch):
        images, _ = batch
        images = images.to(self.device)

        for _ in range(self.steps):
            outputs = self.model(images)
            loss = -(outputs.softmax(1) * outputs.log_softmax(1)).sum(1).mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return self.model(images).detach()

    def reset(self):
        # Tent does not maintain state between batches beyond model parameters
        pass

class FastLanaAdapter(BaseAdapter):
    # Simplified implementation focusing on adapting normalization layers
    def __init__(self, model, config, model_artifacts_path=None):
        super().__init__(model, config, model_artifacts_path)
        self.optimizer = torch.optim.Adam(get_tent_parameters(model), lr=config.get('lr', 1e-4))
        self.model.train()

    def step(self, batch):
        images, _ = batch
        images = images.to(self.device)
        outputs = self.model(images)
        loss = -(outputs.softmax(1) * outputs.log_softmax(1)).sum(1).mean()
        self.optimizer.zero_grad()
        # Fast-LANA is backward-free, this is a simplification
        # A full implementation would use a meta-network
        # Here we mimic adaptation of norm layers via a single backward pass
        loss.backward()
        self.optimizer.step()
        return self.model(images).detach()


# --- LANCE and related classes ---

def fx(x, dt):
    """ state transition function for UKF """
    return x # Simple random walk model for hidden state

def hx(x):
    """ measurement function for UKF """
    return x # We observe the hidden state directly (or a projection)

class UKFController:
    def __init__(self, state_dim, obs_dim, threshold, device):
        self.threshold = threshold
        self.device = device
        points = MerweScaledSigmaPoints(n=state_dim, alpha=.1, beta=2., kappa=1.)
        self.kf = UnscentedKalmanFilter(dim_x=state_dim, dim_z=obs_dim, dt=1., hx=hx, fx=fx, points=points)
        self.kf.x = np.zeros(state_dim)
        self.kf.P *= 0.1 # Initial uncertainty
        self.kf.R *= 0.5 # Measurement noise
        self.kf.Q = np.eye(state_dim) * 0.01 # Process noise
        self.innovation_history = []

    def step(self, observation):
        self.kf.predict()
        self.kf.update(observation)
        
        innovation = self.kf.y
        innovation_mag = np.linalg.norm(innovation)
        self.innovation_history.append(innovation_mag)
        
        running_mean = np.mean(self.innovation_history[-50:])
        running_std = np.std(self.innovation_history[-50:])
        
        is_drift = innovation_mag > (running_mean + self.threshold * running_std)
        return self.kf.x, is_drift, innovation_mag

def collect_state_summary(outputs, model, delta, hooks):
    activations = [hook.get_output() for hook in hooks.values()]
    with torch.no_grad():
        batch_size = outputs.shape[0]
        device = outputs.device
        probs = torch.softmax(outputs, dim=1)
        max_probs, _ = torch.max(probs, dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=1)

        # Use pooled activations for stats
        pooled_activations = torch.cat([act.mean(dim=[2,3]) if act.dim()==4 else act.mean(dim=1) for act in activations], dim=1)

        mu = torch.mean(pooled_activations, dim=-1)
        sigma = torch.std(pooled_activations, dim=-1)

        # Scipy stats are slow, use torch versions or approximations
        skew_val = torch.zeros(batch_size, device=device)
        kurt_val = torch.zeros(batch_size, device=device)
        
        delta_loss_hat = entropy # Approximation
        delta_theta_norm = torch.linalg.norm(delta, dim=1)

        state = torch.stack([
            mu, sigma, skew_val, kurt_val, entropy, max_probs, 
            delta_loss_hat, delta_theta_norm
        ], dim=1)
    return state


class LanceAdapter(BaseAdapter):
    def __init__(self, model, config, model_artifacts_path, ablation_params=None):
        super().__init__(model, config, model_artifacts_path)
        self.ablation_params = ablation_params if ablation_params else {}
        self.params = config['train']['lance_params']

        # Load artifacts
        artifacts = torch.load(model_artifacts_path, map_location=self.device)
        self.total_out_dim = artifacts['total_out_dim']

        # HyperRNN
        hidden_size = self.params.get('rnn_hidden_size', 32)
        self.hyper_rnn = HyperRNN(8, self.total_out_dim, hidden_size).to(self.device)
        self.hyper_rnn.load_state_dict(artifacts['hyper_rnn_state_dict'])
        self.hyper_rnn.eval()

        if self.ablation_params.get('static_hypernet', False):
            self.hyper_rnn.rnn.mode = 'RNN_RELU' # Hack to make it more like a static MLP

        # LowRankMod modules
        self.mod_layers = {}
        self.mod_hooks = {}
        for name, module in self.model.named_modules():
            mod_name = f"lance_mod_{name.replace('.', '_')}"
            if hasattr(self.model, mod_name):
                mod = getattr(self.model, mod_name).to(self.device)
                mod.P.data = artifacts['initial_mod_weights'][name]['P'].to(self.device)
                mod.R.data = artifacts['initial_mod_weights'][name]['R'].to(self.device)
                self.mod_layers[name] = mod
                # Attach forward hooks to collect activations BEFORE the mod layer
                self.mod_hooks[name] = ForwardHook(module)

        # UKF Controller
        self.use_ukf = not self.ablation_params.get('no_ukf', False)
        if self.use_ukf:
            self.ukf = UKFController(
                state_dim=hidden_size, obs_dim=hidden_size, 
                threshold=self.params['ukf_threshold_sigma'], device=self.device
            )
        self.reset()

    def step(self, batch):
        images, _ = batch
        images = images.to(self.device)
        batch_size = images.shape[0]

        if self.hidden_state is None or self.hidden_state.shape[1] != batch_size:
            self.reset_hidden(batch_size)

        with torch.no_grad():
            # 1. Forward pass to get outputs and state
            outputs = self.model(images)
            dummy_delta = torch.zeros(batch_size, self.total_out_dim, device=self.device)
            state_summary = collect_state_summary(outputs, self.model, dummy_delta, self.mod_hooks)
            
            # 2. Hyper-RNN predicts update
            delta, next_hidden = self.hyper_rnn(state_summary, self.hidden_state)

            # 3. UKF drift check
            is_drift, innovation = False, 0
            if self.use_ukf:
                ukf_state, is_drift, innovation = self.ukf.step(next_hidden.squeeze(0).cpu().numpy())
                if is_drift:
                    # Rollback
                    self.rollbacks += 1
                    delta = self.last_safe_delta
                    next_hidden = self.last_safe_hidden
                else:
                    self.last_safe_delta = delta.clone()
                    self.last_safe_hidden = next_hidden.clone()
            
            self.hidden_state = next_hidden

            # 4. Apply low-rank updates (this is a re-computation, but shows the logic)
            def create_hook(delta_slice):
                def hook(module, input, output):
                    mod_name = f"lance_mod_{module._name.replace('.', '_')}"
                    mod_layer = getattr(self.model, mod_name)
                    return mod_layer(output, delta_slice)
                return hook
            
            hooks = []
            current_dim = 0
            for name, mod in self.mod_layers.items():
                target_module = dict(self.model.named_modules())[name]
                target_module._name = name # Attach name for hook
                slice_len = mod.P.numel() + mod.R.numel()
                delta_slice = delta[:, current_dim : current_dim + slice_len]
                hook_handle = target_module.register_forward_hook(create_hook(delta_slice))
                hooks.append(hook_handle)
                current_dim += slice_len

            final_outputs = self.model(images)

            for h in hooks: h.remove()
            return final_outputs, is_drift, innovation

    def reset(self):
        self.hidden_state = None
        self.last_safe_delta = None
        self.last_safe_hidden = None
        self.rollbacks = 0
    
    def reset_hidden(self, batch_size):
        hidden_size = self.hyper_rnn.rnn.hidden_size
        self.hidden_state = torch.zeros(1, batch_size, hidden_size, device=self.device)
        self.last_safe_hidden = torch.zeros(1, batch_size, hidden_size, device=self.device)
        self.last_safe_delta = torch.zeros(batch_size, self.total_out_dim, device=self.device)

class ForwardHook:
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.output = None

    def hook_fn(self, module, input, output):
        self.output = output.detach()

    def get_output(self):
        return self.output

    def close(self):
        self.hook.remove()

# SECTION: METRICS & LOGGING

class MetricsLogger:
    def __init__(self, config, run_name):
        self.config = config
        self.run_name = run_name
        self.results = {}
        self.per_frame_metrics = []
        self.output_dir = config['global_settings']['output_dir']
        os.makedirs(os.path.join(self.output_dir, 'images'), exist_ok=True)
        self.nvm = nvidia_smi.getInstance()
        self.power_log = []

    def start_run(self):
        self.start_time = time.time()
        self.tracker = OfflineEmissionsTracker(country_iso_code="USA", output_dir=self.output_dir)
        self.tracker.start()

    def log_batch(self, outputs, labels, extra_info={}):
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        labels = labels.cpu().numpy()
        correct = (preds == labels).sum()
        accuracy = correct / len(labels)
        self.per_frame_metrics.append({'accuracy': accuracy, **extra_info})
        self.power_log.append(self.nvm.DeviceQuery('power.draw')['gpu'][0]['power_info']['draw'])

    def end_run(self, total_frames):
        self.end_time = time.time()
        self.tracker.stop()
        duration = self.end_time - self.start_time
        
        df = pd.DataFrame(self.per_frame_metrics)
        self.results['mean_accuracy'] = df['accuracy'].mean()
        self.results['std_accuracy'] = df['accuracy'].std()
        self.results['total_frames'] = total_frames
        self.results['duration_sec'] = duration
        self.results['images_per_sec'] = total_frames / duration
        self.results['mean_power_watts'] = np.mean(self.power_log)
        self.results['peak_power_watts'] = np.max(self.power_log)
        
        # Estimate energy
        avg_power = self.results['mean_power_watts']
        self.results['total_energy_joules'] = avg_power * duration
        self.results['energy_per_image_joules'] = self.results['total_energy_joules'] / total_frames

        # CO2 from codecarbon
        emissions_df = pd.read_csv(self.tracker._emissions_datarow.csv_path)
        self.results['co2_eq_grams'] = emissions_df['emissions'].sum() * 1000

        if 'is_drift' in df.columns:
            self.results['total_rollbacks'] = df['is_drift'].sum()

        return self.results

    def save(self):
        filepath = os.path.join(self.output_dir, f"{self.run_name}.json")
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=4)
        print(f"\n--- Results for {self.run_name} ---")
        print(json.dumps(self.results, indent=4))
        print(f"Results saved to {filepath}")
        self.generate_plots()

    def generate_plots(self):
        df = pd.DataFrame(self.per_frame_metrics)
        if 'accuracy' in df.columns:
            plt.figure(figsize=(10, 5))
            plt.plot(df['accuracy'].rolling(window=50).mean())
            plt.title(f'Accuracy (50-frame rolling avg) for {self.run_name}')
            plt.xlabel('Frame')
            plt.ylabel('Accuracy')
            plt.grid(True)
            plt.savefig(os.path.join(self.output_dir, 'images', f'{self.run_name}_accuracy.png'))
            plt.close()

# SECTION: ATTACKS

def craft_pgd_attack(model, images, labels, eps=4/255, alpha=2/255, steps=3):
    images, labels = images.clone().detach(), labels.clone().detach()
    adv_images = images.clone().detach()
    adv_images.requires_grad = True

    for _ in range(steps):
        outputs = model(adv_images)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        grad = torch.autograd.grad(loss, adv_images, retain_graph=False, create_graph=False)[0]
        
        adv_images = adv_images.detach() + alpha * grad.sign()
        delta = torch.clamp(adv_images - images, min=-eps, max=eps)
        adv_images = torch.clamp(images + delta, min=0, max=1).detach()
        adv_images.requires_grad = True

    return adv_images.detach()

# SECTION: MAIN EVALUATION ORCHESTRATOR

def run_experiments(config, dataloaders, model_artifacts_path, all_models):
    """Main orchestrator for running all experiments defined in the config."""
    eval_config = config['evaluate']
    device = torch.device(config['global_settings']['device'])
    
    for exp_name, exp_params in eval_config.items():
        logging.info(f"\n{'='*20} RUNNING EXPERIMENT: {exp_name.upper()} {'='*20}")
        
        for model_name in exp_params['models']:
            for method_spec in exp_params['methods']:
                method_name = method_spec['name'] if isinstance(method_spec, dict) else method_spec
                
                for stream_name in exp_params['streams']:
                    # Multiple seeds
                    for seed in config['global_settings']['seeds']:
                        torch.manual_seed(seed)
                        np.random.seed(seed)

                        run_name = f"{exp_name}_{model_name}_{method_name}_{stream_name}_seed{seed}"
                        logging.info(f"\n----- Starting run: {run_name} -----")

                        # 1. Load model
                        model = all_models[model_name]()
                        model.to(device)
                        model.eval()

                        # 2. Instantiate Adapter
                        adapter = None
                        if method_name == 'Source-only':
                            adapter = SourceOnlyAdapter(model, config)
                        elif method_name == 'Tent':
                            adapter = TentAdapter(model, method_spec['params'])
                        elif method_name == 'BN-adapt':
                            adapter = BNAdaptAdapter(model, config)
                        elif method_name == 'FAST-LANA':
                            adapter = FastLanaAdapter(model, method_spec['params'])
                        elif 'LANCE' in method_name:
                            ablation_params = method_spec.get('params', {})
                            adapter = LanceAdapter(model, config, model_artifacts_path, ablation_params)
                        else:
                            logging.warning(f"Adapter '{method_name}' not implemented. Skipping.")
                            continue

                        # 3. Run evaluation loop
                        logger = MetricsLogger(config, run_name)
                        logger.start_run()
                        
                        dataloader = dataloaders[stream_name]
                        total_frames = 0
                        pbar = tqdm(dataloader, desc=run_name)

                        use_pgd = exp_params.get('use_pgd', False)

                        for i, (images, labels) in enumerate(pbar):
                            if exp_params.get('max_frames') and total_frames >= exp_params['max_frames']:
                                break
                            
                            images, labels = images.to(device), labels.to(device)
                            
                            if use_pgd and i % exp_params['pgd_interval'] < exp_params['pgd_duration']:
                                images = craft_pgd_attack(adapter.model, images, labels)
                            
                            # Adapter step
                            adapter_output = adapter.step((images, labels))
                            
                            # Unpack output
                            if isinstance(adapter_output, tuple):
                                outputs, is_drift, innovation = adapter_output
                                extra_info = {'is_drift': is_drift, 'innovation': innovation}
                            else:
                                outputs = adapter_output
                                extra_info = {}
                            
                            logger.log_batch(outputs, labels, extra_info)
                            total_frames += images.shape[0]

                        # 4. Finalize and save results
                        logger.end_run(total_frames)
                        logger.save()

    logging.info("All experiments finished.")
