import random
import numpy as np
import torch
import torch.nn as nn
import math
from collections import deque

import gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from IPython.display import clear_output
import time
import pygame
from PIL import Image
from matplotlib import cm
from gym.core import ObservationWrapper
from gym.spaces import Box

import mlflow
from mlflow.models import infer_signature
import atari_wrappers
import utils
from framebuffer import FrameBuffer

import mlflow
from mlflow.models import infer_signature

class PreprocessAtariObs(ObservationWrapper):
    def __init__(self, env):
        ObservationWrapper.__init__(self, env)

        self.img_size = (1, 64, 64)
        self.observation_space = Box(0.0, 1.0, self.img_size)


    def _to_gray_scale(self, rgb, channel_weights=[0.8, 0.1, 0.1]):
        return rgb[:,:,0] * channel_weights[0] + rgb[:,:,1] * channel_weights[1] + rgb[:,:,2] * channel_weights[2]


    def observation(self, img):
        img = Image.fromarray(img)
        img = img.crop((6, 29, 160 - 6, 210))
        img = img.resize((64, 64))
        img = np.array(img).astype(np.float32)
        img = (img - img.min()) / (img.max() - img.min())
        img = self._to_gray_scale(img)
        return img[np.newaxis, :]

def PrimaryAtariWrap(env, clip_rewards=True):

    env = atari_wrappers.MaxAndSkipEnv(env, skip=4)
    env = atari_wrappers.EpisodicLifeEnv(env)
    env = atari_wrappers.FireResetEnv(env)

    if clip_rewards:
        env = atari_wrappers.ClipRewardEnv(env)

    env = PreprocessAtariObs(env)
    return env

def make_env(clip_rewards=True, seed=None):
    ENV_NAME = "BreakoutNoFrameskip-v4"
    env = gym.make(ENV_NAME)  # create raw env
    if seed is not None:
        env.seed(seed)
    env = PrimaryAtariWrap(env, clip_rewards)
    env = FrameBuffer(env, n_frames=4, dim_order='pytorch')
    return env


class DQNAgent(nn.Module):
    def __init__(self, state_shape, n_actions, epsilon=0):

        super().__init__()
        self.epsilon = epsilon
        self.n_actions = n_actions
        self.state_shape = state_shape
        
        self.conv1 = nn.Conv2d(in_channels=state_shape[0], out_channels=16, kernel_size=3, stride=2)
        self.ReLU1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2)
        self.ReLU2 = nn.ReLU()
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2)
        self.ReLU3 = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fe = nn.Sequential(self.conv1, self.ReLU1, self.conv2, self.ReLU2,
                                 self.conv3, self.ReLU3, self.flatten)
        
        self.fc_V = nn.Linear(49 * 64, 1024)
        self.fc_V_2 = nn.Linear(1024, 1)
        self.ReLuV = nn.ReLU()
        
        self.fc_A = nn.Linear(49 * 64, 1024)
        self.fc_A_2 = nn.Linear(1024, n_actions)
        self.ReLuA = nn.ReLU()
        
    def forward(self, state_t):

        features = self.fe(state_t)
        
        V = self.fc_V(features)
        V = self.ReLuV(V)
        V = self.fc_V_2(V)
        
        A = self.fc_A(features)
        A = self.ReLuA(A)
        A = self.fc_A_2(A)
        
        qvalues = V + A - torch.mean(A, dim=1, keepdim=True)

        return qvalues

    def get_qvalues(self, states):

        model_device = next(self.parameters()).device
        states = torch.tensor(states, device=model_device, dtype=torch.float32)
        qvalues = self.forward(states)
        return qvalues.data.cpu().numpy()

    def sample_actions(self, qvalues):

        epsilon = self.epsilon
        batch_size, n_actions = qvalues.shape

        random_actions = np.random.choice(n_actions, size=batch_size)
        best_actions = qvalues.argmax(axis=-1)

        should_explore = np.random.choice(
            [0, 1], batch_size, p=[1-epsilon, epsilon])
        return np.where(should_explore, random_actions, best_actions)


def evaluate(env, agent, n_games=1, greedy=False, t_max=10000):
    rewards = []
    for _ in range(n_games):
        s = env.reset()
        reward = 0
        for _ in range(t_max):
            qvalues = agent.get_qvalues([s])
            action = qvalues.argmax(axis=-1)[0] if greedy else agent.sample_actions(qvalues)[0]
            s, r, done, _ = env.step(action)
            reward += r
            if done:
                break

        rewards.append(reward)
    return np.mean(rewards)

class ReplayBuffer(object):
    def __init__(self, size):
        self._storage = deque()
        self._maxsize = size


    def __len__(self):
        return len(self._storage)

    def add(self, obs_t, action, reward, obs_tp1, done):

        data = (obs_t, action, reward, obs_tp1, done)

        if self.__len__() == self._maxsize:
            self._storage.popleft()
            
        self._storage.append(data)

    def sample(self, batch_size):

        idxes = np.random.choice(np.arange(self.__len__()), batch_size)

        obs_batch = np.array([self._storage[i][0] for i in idxes])
        act_batch = np.array([self._storage[i][1] for i in idxes])
        rew_batch = np.array([self._storage[i][2] for i in idxes])
        next_obs_batch = np.array([self._storage[i][3] for i in idxes])
        done_mask = np.array([self._storage[i][4] for i in idxes])
        
        return obs_batch, act_batch, rew_batch, next_obs_batch, done_mask

def play_and_record(initial_state, agent, env, exp_replay, n_steps=1):

    s = initial_state
    sum_rewards = 0
    for i in range(n_steps):
        qvalues = agent.get_qvalues([s])
        action = agent.sample_actions(qvalues)[0]
        next_s, r, done, _ = env.step(action)
        sum_rewards += r
        exp_replay.add(s, action, r, next_s, done)
        if done:
            s = env.reset()
        else:
            s = next_s
        
    return sum_rewards, s


def compute_td_loss(states, actions, rewards, next_states, is_done,
                    agent, target_network,
                    gamma=0.99,
                    check_shapes=False,
                    device='cpu'):
    states = torch.tensor(states, device=device, dtype=torch.float32) 
    actions = torch.tensor(actions, device=device, dtype=torch.int64) 
    rewards = torch.tensor(rewards, device=device, dtype=torch.float32) 
    next_states = torch.tensor(next_states, device=device, dtype=torch.float)
    is_done = torch.tensor(
        is_done.astype('float32'),
        device=device,
        dtype=torch.float32,
    )  
    is_not_done = 1 - is_done

    predicted_qvalues = agent(states)

    predicted_next_qvalues = target_network(next_states) 
    
    predicted_qvalues_for_actions = predicted_qvalues[range(len(actions)), actions] 

    qvalues_argmax = torch.argmax(predicted_qvalues.detach(), dim=1)
    next_state_values = predicted_next_qvalues[range(len(actions)), qvalues_argmax]

    target_qvalues_for_actions = rewards + gamma * next_state_values
    target_qvalues_for_actions = torch.where(is_not_done.type(torch.bool), target_qvalues_for_actions, rewards)

    loss = torch.mean((predicted_qvalues_for_actions - target_qvalues_for_actions.detach()) ** 2)

    return loss

def wait_for_keyboard_interrupt():
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass


def main():
    ENV_NAME = "BreakoutNoFrameskip-v4"
    env = gym.make(ENV_NAME)
    env.reset()

    n_cols = 5
    n_rows = 2
    fig = plt.figure(figsize=(16, 9))

    for row in range(n_rows):
        for col in range(n_cols):
            ax = fig.add_subplot(n_rows, n_cols, row * n_cols + col + 1)
            ax.imshow(env.render('rgb_array'))
            env.step(env.action_space.sample())
    plt.show()
    
    state = env.step(env.action_space.sample())[0]
    print(state.shape)

    im = Image.fromarray(state)
    im = im.crop((0, 29, 0, 210))
    im = im.resize((64, 64))
    im_n = np.array(im)
    print(im_n[:, : , 0].shape)

    env = gym.make(ENV_NAME)  
    env = PreprocessAtariObs(env)
    observation_shape = env.observation_space.shape
    n_actions = env.action_space.n
    env.reset()
    obs, _, _, _ = env.step(env.action_space.sample())
    
    n_cols = 5
    n_rows = 2
    fig = plt.figure(figsize=(16, 9))
    obs = env.reset()
    for row in range(n_rows):
        for col in range(n_cols):
            ax = fig.add_subplot(n_rows, n_cols, row * n_cols + col + 1)
            ax.imshow(obs[0, :, :], interpolation='none', cmap='gray')
            obs, _, _, _ = env.step(env.action_space.sample())
    plt.show()

    env = make_env()
    env.reset()
    n_actions = env.action_space.n
    state_shape = env.observation_space.shape

    for _ in range(12):
        obs, _, _, _ = env.step(env.action_space.sample())
    
    plt.figure(figsize=[12,10])
    plt.title("Game image")
    plt.imshow(env.render("rgb_array"))
    plt.show()
    
    plt.figure(figsize=[15,15])
    plt.title("Agent observation (4 frames top to bottom)")
    plt.imshow(utils.img_by_obs(obs, state_shape), cmap='gray')
    plt.show()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    a = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=3, stride=2)
    b = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2)
    c = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2)
    net = nn.Sequential(a,b,c).to(device)
    s = env.reset()
    states = torch.tensor([s], device=device, dtype=torch.float32).to(device)
    print(net(states).shape)
    
    agent = DQNAgent(state_shape, n_actions, epsilon=0.5).to(device)
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    env = make_env(seed)
    state_shape = env.observation_space.shape
    n_actions = env.action_space.n
    state = env.reset()
    
    agent = DQNAgent(state_shape, n_actions, epsilon=1).to(device)
    target_network = DQNAgent(state_shape, n_actions).to(device)
    target_network.load_state_dict(agent.state_dict())

    REPLAY_BUFFER_SIZE = 10**4
    N_STEPS = 100
    
    exp_replay = ReplayBuffer(REPLAY_BUFFER_SIZE)
    for i in trange(REPLAY_BUFFER_SIZE // N_STEPS):
        play_and_record(state, agent, env, exp_replay, n_steps=N_STEPS)
        if len(exp_replay) == REPLAY_BUFFER_SIZE:
            break
    print(len(exp_replay))
    
    timesteps_per_epoch = 1
    batch_size = 16
    total_steps = 4 * 10**5
    decay_steps = 10**4
    
    opt = torch.optim.Adam(agent.parameters(), lr=1e-4)
    
    init_epsilon = 1
    final_epsilon = 0.1
    
    loss_freq = 50
    refresh_target_network_freq = 5000
    eval_freq = 5000
    
    max_grad_norm = 50
    
    n_lives = 5

    mean_rw_history = []
    td_loss_history = []
    grad_norm_history = []
    initial_state_v_history = []
    
    step = 0


    state = env.reset()
    with trange(step, total_steps + 1) as progress_bar:
        for step in progress_bar:
            if not utils.is_enough_ram():
                print('less that 100 Mb RAM available, freezing')
                print('make sure everything is ok and use KeyboardInterrupt to continue')
                wait_for_keyboard_interrupt()
    
            agent.epsilon = utils.linear_decay(init_epsilon, final_epsilon, step, decay_steps)
    
            # play
            _, state = play_and_record(state, agent, env, exp_replay, timesteps_per_epoch)
    
            # train
            batch = exp_replay.sample(batch_size)
            loss = compute_td_loss(*batch, agent, target_network)
    
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
            opt.step()
            opt.zero_grad()
    
            if step % loss_freq == 0:
                td_loss_history.append(loss.data.cpu().item())
                grad_norm_history.append(grad_norm.cpu())
    
            if step % refresh_target_network_freq == 0:
                target_network.load_state_dict(agent.state_dict())
    
            if step % eval_freq == 0:
                mean_rw_history.append(evaluate(
                    make_env(clip_rewards=True, seed=step), agent, n_games=3 * n_lives, greedy=True)
                )
                initial_state_q_values = agent.get_qvalues(
                    [make_env(seed=step).reset()]
                )
                initial_state_v_history.append(np.max(initial_state_q_values))
    
                clear_output(True)
                print("buffer size = %i, epsilon = %.5f" %
                    (len(exp_replay), agent.epsilon))
    
                plt.figure(figsize=[16, 9])
    
                plt.subplot(2, 2, 1)
                plt.title("Mean reward per life")
                plt.plot(mean_rw_history)
                plt.grid()
    
                plt.subplot(2, 2, 2)
                plt.title("TD loss history (smoothened)")
                plt.plot(utils.smoothen(td_loss_history))
                plt.grid()
    
                plt.subplot(2, 2, 3)
                plt.title("Initial state V")
                plt.plot(initial_state_v_history)
                plt.grid()
    
                plt.subplot(2, 2, 4)
                plt.title("Grad norm history (smoothened)")
                plt.plot(utils.smoothen(grad_norm_history))
                plt.grid()
       
                plt.show()

    final_score = evaluate(make_env(clip_rewards=False, seed=9), agent, n_games=30, greedy=True, t_max=10 * 1000)
    print('final score:', final_score)
    torch.save(agent.state_dict(), 'agent_breakout.pt')
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(1, 1, 1)
    
    eval_env_clipped = make_env(clip_rewards=True)
    record_clipped = utils.play_and_log_episode(eval_env_clipped, agent)
    
    ax.scatter(record_clipped['v_mc'], record_clipped['v_agent'])
    ax.plot(sorted(record_clipped['v_mc']), sorted(record_clipped['v_mc']),
           'black', linestyle='--', label='x=y')
    
    ax.grid()
    ax.legend()
    ax.set_title('State Value Estimates')
    ax.set_xlabel('Monte-Carlo')
    ax.set_ylabel('Agent')
    
    fig1, ax1 = plt.subplots()
    ax1.plot(mean_rw_history)
    ax1.set_title("Mean reward per life")
    ax1.grid()
    
    fig2, ax2 = plt.subplots()
    ax2.plot(utils.smoothen(td_loss_history))
    ax2.set_title("TD loss history (smoothened)")
    ax2.grid()
    
    fig3, ax3 = plt.subplots()
    ax3.plot(initial_state_v_history)
    ax3.set_title("Initial state V")
    ax3.grid()
    
    fig4, ax4 = plt.subplots()
    ax4.plot(utils.smoothen(grad_norm_history))
    ax4.set_title("Grad norm history (smoothened)")
    ax4.grid()

    mlflow.set_tracking_uri('file:///C:/Users/Seva/Desktop/%D0%9D%D0%BE%D0%B2%D0%B0%D1%8F%20%D0%BF%D0%B0%D0%BF%D0%BA%D0%B0%20%282%29/mlruns/0/5423995c46ab4605931a11531a0ba276/artifacts')
    mlflow.set_experiment("MLflow Breakout")
                            
    with mlflow.start_run() as run:
        mlflow.log_metrics({"total_reward": total_reward})
        mlflow.log_figure(fig, "State Value Estimates.png")
        mlflow.log_figure(fig1, "Mean reward per life.png")
        mlflow.log_figure(fig2, "TD loss history (smoothened).png")
        mlflow.log_figure(fig3, "Initial state V.png")
        mlflow.log_figure(fig4, "Grad norm history (smoothened).png")


main()
    