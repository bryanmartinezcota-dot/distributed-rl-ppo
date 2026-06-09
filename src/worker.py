import ray
import torch
import gym
import numpy as np
from src.models import ActorCritic

@ray.remote
class RolloutWorker:
    def __init__(self, env_name, state_dim, action_dim):
        self.env = gym.make(env_name)
        self.policy = ActorCritic(state_dim, action_dim)
        
    def set_weights(self, weights):
        """Update the local policy with global weights."""
        self.policy.load_state_dict(weights)
        
    def collect_trajectory(self, max_steps=2048):
        """Run the environment and collect (s, a, r, s') transitions."""
        states, actions, rewards, log_probs = [], [], [], []
        state = self.env.reset()[0]
        
        with torch.no_grad():
            for _ in range(max_steps):
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                action, log_prob, _, _ = self.policy.get_action_and_value(state_tensor)
                
                action_np = action.squeeze(0).numpy()
                next_state, reward, done, _, _ = self.env.step(action_np)
                
                states.append(state)
                actions.append(action_np)
                rewards.append(reward)
                log_probs.append(log_prob.item())
                
                state = next_state
                if done:
                    state = self.env.reset()[0]
                    
        return {
            "states": np.array(states),
            "actions": np.array(actions),
            "rewards": np.array(rewards),
            "log_probs": np.array(log_probs)
        }
