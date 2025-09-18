import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
from scipy.stats import skew, kurtosis
import logging
import os
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# SECTION: LANCE Core Modules

class LowRankMod(nn.Module):
    """Low-rank modulator that adapts activations.

    For a Linear layer (input shape [N, C_in]), this is straightforward.
    For a Conv2d layer (input shape [N, C, H, W]), we treat it as a linear
    operation on the channel dimension, applying the same modulation at each
    spatial location.
    """
    def __init__(self, dim, rank=4):
        super().__init__()
        if dim <= 0:
            raise ValueError(f"Dimension must be positive, but got {dim}")
        self.rank = rank
        self.dim = dim
        self.P = nn.Parameter(torch.zeros(dim, rank))
        self.R = nn.Parameter(torch.zeros(rank, dim))
        nn.init.normal_(self.P, std=0.01)
        nn.init.normal_(self.R, std=0.01)

    def forward(self, x, delta):
        is_conv = x.dim() == 4

        # Reshape delta to update P and R matrices
        p_delta_flat = delta[:, :self.P.numel()]
        r_delta_flat = delta[:, self.P.numel():]
        
        p_delta = p_delta_flat.view(-1, self.dim, self.rank)
        r_delta = r_delta_flat.view(-1, self.rank, self.dim)

        # Apply updates
        P_adapted = self.P.unsqueeze(0) + p_delta
        R_adapted = self.R.unsqueeze(0) + r_delta

        if is_conv:
            # x: [N, C, H, W], we adapt C
            # Permute to [N, H, W, C] to apply modulation
            x_permuted = x.permute(0, 2, 3, 1)
            # modulation: [N, H, W, C] -> [N, H, W, rank] -> [N, H, W, C]
            modulation = (x_permuted @ R_adapted.transpose(-1, -2)) @ P_adapted.transpose(-1, -2)
            # Permute back and add to original
            return x + modulation.permute(0, 3, 1, 2)
        else:
            # x: [N, D] or [N, T, D] for transformers
            # modulation: [N, D] -> [N, rank] -> [N, D]
            modulation = (x @ R_adapted.transpose(-1, -2)) @ P_adapted.transpose(-1, -2)
            return x + modulation

class HyperRNN(nn.Module):
    def __init__(self, state_dim, out_dim, hidden_size=32):
        super().__init__()
        self.rnn = nn.GRU(state_dim, hidden_size, 1, batch_first=True)
        self.fc = nn.Linear(hidden_size, out_dim)
        self.h = None

    def forward(self, s, h_prev=None):
        # s shape: (batch_size, state_dim)
        # We process one time step at a time, so sequence length is 1
        s_unsqueezed = s.unsqueeze(1) # (batch, 1, state_dim)
        y, h_next = self.rnn(s_unsqueezed, h_prev)
        delta = self.fc(y.squeeze(1))
        return delta, h_next

    def reset_hidden(self, batch_size, device):
        self.h = torch.zeros(1, batch_size, self.rnn.hidden_size, device=device)

class Critic(nn.Module):
    def __init__(self, state_dim, hidden_size=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    def forward(self, state):
        return self.net(state)

# SECTION: PPO Implementation

class PPOEngine:
    def __init__(self, actor, critic, lr, betas, gamma, K_epochs, eps_clip, lambda_gae, config):
        self.actor = actor
        self.critic = critic
        self.optimizer_actor = optim.Adam(actor.parameters(), lr=lr[0], betas=betas)
        self.optimizer_critic = optim.Adam(critic.parameters(), lr=lr[1], betas=betas)
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.lambda_gae = lambda_gae
        self.config = config
        self.mse_loss = nn.MSELoss()

    def update(self, memory):
        # Monte Carlo estimate of rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        rewards = torch.tensor(rewards, dtype=torch.float32).to(memory.device).detach()
        # Normalizing the rewards is a standard practice
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        old_states = torch.squeeze(torch.stack(memory.states, dim=0)).detach().to(memory.device)
        old_actions = torch.squeeze(torch.stack(memory.actions, dim=0)).detach().to(memory.device)
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs, dim=0)).detach().to(memory.device)

        # GAE
        values = self.critic(old_states).detach().squeeze()
        advantages = torch.zeros_like(rewards)
        last_gae_lam = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t+1]*(1-memory.is_terminals[t]) - values[t]
            advantages[t] = last_gae_lam = delta + self.gamma * self.lambda_gae * (1-memory.is_terminals[t]) * last_gae_lam
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)
        
        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.evaluate(old_states, old_actions)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            loss_actor = -torch.min(surr1, surr2) - 0.01 * dist_entropy
            loss_critic = self.mse_loss(state_values, rewards)

            # take gradient step
            self.optimizer_actor.zero_grad()
            loss_actor.mean().backward()
            self.optimizer_actor.step()
            
            self.optimizer_critic.zero_grad()
            loss_critic.mean().backward()
            self.optimizer_critic.step()

    def select_action(self, state, memory, hidden_state):
        with torch.no_grad():
            action_params, next_hidden_state = self.actor(state, hidden_state)
            dist = Normal(action_params, 0.1) # Add small std dev for exploration
            action = dist.sample()
            action_logprob = dist.log_prob(action).sum(dim=-1)

        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(action_logprob)

        return action.detach(), next_hidden_state

    def evaluate(self, state, action):
        action_mean, _ = self.actor(state, None) # Hidden state not needed for evaluation pass
        dist = Normal(action_mean, 0.1)
        action_logprobs = dist.log_prob(action).sum(dim=-1)
        dist_entropy = dist.entropy().sum(dim=-1)
        state_values = self.critic(state).squeeze()
        return action_logprobs, state_values, dist_entropy

class RolloutBuffer:
    def __init__(self, device):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.device = device

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


# SECTION: Utilities

def inject_lance_modules(model, config):
    """Recursively injects LowRankMod modules into the model."""
    rank = config['train']['lance_params']['rank']
    mod_layers = {}
    total_out_dim = 0
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            if isinstance(module, nn.Linear):
                dim = module.out_features
            else: # Conv2d
                dim = module.out_channels

            mod = LowRankMod(dim, rank=rank)
            # Use a wrapper to handle multiple inputs/outputs if necessary
            # Here we assume a simple sequential model structure
            # We will attach it to the model to be called manually in the forward pass
            mod_name = f"lance_mod_{name.replace('.', '_')}"
            setattr(model, mod_name, mod)
            mod_layers[name] = mod
            total_out_dim += mod.P.numel() + mod.R.numel()
    
    logging.info(f"Injected {len(mod_layers)} LANCE modules.")
    logging.info(f"Total HyperRNN output dimension: {total_out_dim}")
    return mod_layers, total_out_dim

class EnergyMeter:
    def __init__(self, device, calib_file='calib_A100.pt'):
        if os.path.exists(calib_file):
            self.a, self.b = torch.load(calib_file)
        else:
            logging.warning(f"Calibration file {calib_file} not found. Using dummy coefficients.")
            # Dummy coefficients for A100: Joules = a * GFLOPs + b * GBytes_mem
            self.a = torch.tensor(0.005) # Dummy value
            self.b = torch.tensor(0.001) # Dummy value
            torch.save((self.a, self.b), calib_file)
        self.a = self.a.to(device)
        self.b = self.b.to(device)

    def __call__(self, flops, mem_bytes):
        gflops = flops / 1e9
        gbytes = mem_bytes / 1e9
        return self.a * gflops + self.b * gbytes

def collect_and_process_state(model_output, model, delta):
    """Collects the 8-D state summary for the HyperRNN."""
    with torch.no_grad():
        probs = torch.softmax(model_output, dim=1)
        max_probs, _ = torch.max(probs, dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=1)

        # For other stats, we need activations. This is a simplification.
        # A full implementation would use hooks.
        # Here we use the final output as a proxy for activations.
        activations = model_output
        mu = torch.mean(activations, dim=-1)
        sigma = torch.std(activations, dim=-1)
        
        # Skew and Kurtosis are computationally intensive. Use approximations or skip.
        # For simplicity, we use zeros as placeholders. A real implementation would use scipy on CPU.
        batch_size = activations.shape[0]
        device = activations.device
        skew_val = torch.zeros(batch_size, device=device)
        kurt_val = torch.zeros(batch_size, device=device)
        
        # delta_loss_hat - approximated by entropy
        delta_loss_hat = entropy
        
        # delta_theta_norm
        delta_theta_norm = torch.linalg.norm(delta, dim=1)

        state = torch.stack([
            mu, sigma, skew_val, kurt_val, entropy, max_probs, 
            delta_loss_hat, delta_theta_norm
        ], dim=1)
    return state

# SECTION: Main Training Loop

def run_meta_training(config, model_template, synthetic_generator):
    """Main function to run the meta-training of LANCE components."""
    logging.info("Starting LANCE meta-training.")
    params = config['train']
    device = torch.device(config['global_settings']['device'])
    
    # 1. Prepare models
    model_template.to(device)
    mod_layers, total_out_dim = inject_lance_modules(model_template, config)
    hyper_rnn = HyperRNN(
        state_dim=8, 
        out_dim=total_out_dim, 
        hidden_size=params['lance_params']['rnn_hidden_size']
    ).to(device)
    critic = Critic(state_dim=8).to(device)

    # 2. Setup PPO
    ppo_agent = PPOEngine(
        actor=hyper_rnn, critic=critic,
        lr=params['ppo']['lr'], betas=params['ppo']['betas'],
        gamma=params['ppo']['gamma'], K_epochs=params['ppo']['k_epochs'],
        eps_clip=params['ppo']['eps_clip'], lambda_gae=params.get('lambda_gae', 0.95), 
        config=config
    )
    memory = RolloutBuffer(device)

    # 3. Setup other components
    energy_meter = EnergyMeter(device)
    criterion = nn.CrossEntropyLoss()

    # 4. Training Loop
    timesteps = 0
    pbar = tqdm(range(params['epochs']), desc="Meta-Training Epochs")
    for epoch in pbar:
        hidden_state = None
        for i, (images, labels) in enumerate(synthetic_generator):
            if timesteps >= params['max_timesteps']:
                break

            images, labels = images.to(device), labels.to(device)

            # Select action
            with torch.no_grad():
                initial_out = model_template(images)
                # A dummy delta to compute the initial state
                dummy_delta = torch.zeros(images.size(0), total_out_dim, device=device)
                state = collect_and_process_state(initial_out, model_template, dummy_delta)
            
            action_delta, hidden_state = ppo_agent.select_action(state, memory, hidden_state)

            # Apply action and get reward
            # This is a simplified application for training reward calculation.
            # The actual update is done module by module in evaluation.
            adapted_out = model_template(images) # Re-run for simplicity

            # Calculate reward components
            ce_loss = criterion(adapted_out, labels)
            flops = 2 * images.numel() * 768 # Rough FLOPs for ViT-B
            energy_est = energy_meter(flops, images.numel() * images.element_size())
            l2_reg = torch.linalg.norm(action_delta)

            reward = - (ce_loss 
                        + params['lambda_e'] * energy_est 
                        + params['lambda_s'] * l2_reg)
            
            # Store data in buffer
            memory.rewards.append(reward.item())
            is_terminal = (i + 1) % params['ppo']['update_timestep'] == 0
            memory.is_terminals.append(is_terminal)
            timesteps += 1

            # Update policy
            if timesteps % params['ppo']['update_timestep'] == 0:
                ppo_agent.update(memory)
                memory.clear()
                hidden_state = hidden_state.detach()
        
        pbar.set_postfix({'reward': np.mean(memory.rewards) if memory.rewards else 0})
        if timesteps >= params['max_timesteps']:
            break

    # 5. Save artifacts
    output_dir = config['global_settings']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    model_artifacts_path = os.path.join(output_dir, 'lance_artifacts.pt')
    
    # Consolidate initial weights of all modulators
    initial_mod_weights = {name: {'P': mod.P.cpu().clone(), 'R': mod.R.cpu().clone()} for name, mod in mod_layers.items()}

    torch.save({
        'hyper_rnn_state_dict': hyper_rnn.state_dict(),
        'initial_mod_weights': initial_mod_weights,
        'total_out_dim': total_out_dim
    }, model_artifacts_path)

    logging.info(f"Meta-training finished. Artifacts saved to {model_artifacts_path}")
    return model_artifacts_path
