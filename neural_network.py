import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.distributions.categorical import Categorical
import plotly.graph_objects as go
import logging

DEVICE = 'cpu'

# Policy and value model
class ActorCriticNetwork(nn.Module):
  def __init__(self, obs_space_size, action_space_size):
    super().__init__()

    self.shared_layers = nn.Sequential(
        nn.Linear(obs_space_size, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU())
    
    self.policy_layers = nn.Sequential(
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, action_space_size))
    
    self.value_layers = nn.Sequential(
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 1))
    
  def value(self, obs):
    shared_layers = self.shared_layers(obs)
    value = self.value_layers(shared_layers)
    return value

  def policy(self, obs):
    shared_layers = self.shared_layers(obs)
    policy_logits = self.policy_layers(shared_layers)
    return policy_logits

  def forward(self, obs): # forward propagation
    shared_layers = self.shared_layers(obs)
    policy_logits = self.policy_layers(shared_layers)
    value = self.value_layers(shared_layers)
    return policy_logits, value
  
class PPOTrainer():
   def __init__(self, actor_critic, ppo_clip_value=0.2, target_kl_divergence=0.01, max_policy_train_iterations=80, value_train_iterations=80, policy_lr=3e-4, value_lr=1e-2):
      self.ac = actor_critic
      self.ppo_clip_value = ppo_clip_value
      self.target_kl_divergence = target_kl_divergence
      self.max_policy_train_iterations = max_policy_train_iterations
      self.value_train_iterations = value_train_iterations

      policy_params = list(self.ac.shared_layers.parameters()) + \
      list(self.ac.policy_layers.parameters())
      self.policy_optim = optim.Adam(policy_params, lr=policy_lr)
      
      value_params = list(self.ac.shared_layers.parameters()) + \
        list(self.ac.policy_layers.parameters())
      self.value_optim = optim.Adam(value_params, lr=value_lr)

   def train_policy(self, obs, acts, old_log_probs, generalized_advantage_estimation):
    for _ in range(self.max_policy_train_iterations):
      self.policy_optim.zero_grad()

      new_logits = self.ac.policy(obs)
      new_logits = Categorical(logits=new_logits)
      new_log_probs = new_logits.log_prob(acts)
      policy_ratio = torch.exp(new_log_probs) / torch.exp(old_log_probs)

      clippings = policy_ratio.clamp(
          1 - self.ppo_clip_value, 1 + self.ppo_clip_value
      )
      
      clipped_cost = clippings * generalized_advantage_estimation
      full_cost = policy_ratio * generalized_advantage_estimation
      policy_cost = -torch.min(full_cost, clipped_cost).mean()
      policy_cost.backward() # backpropagation
      self.policy_optim.step()

      kl_divergence = (old_log_probs - new_log_probs).mean()
      if kl_divergence >= self.target_kl_divergence:
        break

   def train_value(self, obs, returns):
    for _ in range(self.value_train_iterations):
        self.value_optim.zero_grad()
        values = self.ac.value(obs)
        value_cost = (returns - values) ** 2 # residual; finding cost
        value_cost = value_cost.mean() # see MSE formula
        value_cost.backward() # backpropagation
        self.value_optim.step()
  

def discount_rewards(rewards, gamma=0.99):
   new_rewards = [float(rewards[-1])]
   for i in reversed(range(len(rewards)-1)):
      new_rewards.append(float(rewards[i]) + gamma * new_rewards[-1])
   return np.array(new_rewards[::-1])

def calculate_generalized_advantage_estimation(rewards, values, gamma=0.99, decay=0.99): # see GAE paper
   next_values = np.concatenate([values[1:], [0]])
   print("values: " + str(values))
   print("next values: " + str(next_values))
   deltas = [new + gamma * next_val - val for new, val, next_val in zip(rewards, values, next_values)]
   print(deltas)
   generalized_advantage_estimations = [deltas[-1]]
   for i in reversed(range(len(deltas)-1)):
      generalized_advantage_estimations.append(deltas[i] + decay * gamma * generalized_advantage_estimations[-1])
   return np.array(generalized_advantage_estimations[::-1])

def rollout(model, env, max_steps=1000):
    train_data = [[], [], [], [], []] # obs, act, reward, values, act_log_probs
    obs, _ = env.reset()

    ep_reward = 0
    for _ in range(max_steps):
        logits, val = model(torch.tensor([obs], dtype=torch.float32, device=DEVICE))
        act_distribution = Categorical(logits=logits)
        act = act_distribution.sample()
        act_log_prob = act_distribution.log_prob(act).item()

        act, val = act.item(), val.item()

        next_obs, reward, done, _, _, = env.step(act)

        for i, item in enumerate((obs, act, reward, val, act_log_prob)):
            train_data[i].append(item)

        obs = next_obs
        ep_reward += reward
        if done:
            break

    train_data = [np.asarray(x) for x in train_data]
    train_data[3] = calculate_generalized_advantage_estimation(train_data[2], train_data[3])
    return train_data, ep_reward

env = gym.make("CartPole-v1")
model = ActorCriticNetwork(env.observation_space.shape[0], env.action_space.n)
model = model.to(DEVICE)
train_data, reward = rollout(model, env)

# Defining training params
n_episodes = 500
print_freq = 20

ppo = PPOTrainer(
   model,
   policy_lr = 3e-4,
   value_lr = 1e-3,
   target_kl_divergence = 0.02,
   max_policy_train_iterations = 40,
   value_train_iterations = 40
)

#Training loop
ep_rewards = []
for episode_idx in range(n_episodes): 

    train_data, reward = rollout(model, env)
    print("rewards: " + str(reward))
    ep_rewards.append(reward)

    permute_idxs = np.random.permutation(len(train_data[0]))

    obs = torch.tensor(train_data[0][permute_idxs], dtype=torch.float32, device=DEVICE)
    acts = torch.tensor(train_data[1][permute_idxs], dtype=torch.int32, device=DEVICE)
    generalized_advantage_estimation = torch.tensor(train_data[3][permute_idxs], dtype=torch.float32, device=DEVICE)
    act_log_probs = torch.tensor(train_data[4][permute_idxs], dtype=torch.float32, device=DEVICE)   

    returns = discount_rewards(train_data[2][permute_idxs])
    returns = torch.tensor(returns, dtype=torch.float32, device=DEVICE)

    ppo.train_policy(obs, acts, act_log_probs, generalized_advantage_estimation)
    ppo.train_value(obs, returns)

    if (episode_idx + 1) % print_freq == 0:
        print('Episode #: [{}] | Average Reward #: [{:.1f}]'.format(
        episode_idx + 1, np.mean(ep_rewards[-print_freq:])))

