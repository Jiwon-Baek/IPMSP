import torch
import numpy as np
import pandas as pd
import os
from agent.ppo import PPO
from environment.env import PMSP
from cfg_local import Configure


def tround(t, decimal):
    f = t.clone()
    f = f.tolist()
    return [round(f[i], decimal) for i in range(len(f))]


def nround(t, decimal):
    f = t.tolist()
    return [round(f[i], decimal) for i in range(len(f))]


cfg = Configure()
weight_tard = 0.5
weight_setup = 1 - weight_tard
optim = cfg.optim
learning_rate = cfg.lr
K_epoch = cfg.K_epoch
T_horizon = cfg.T_horizon
num_episode = cfg.n_episode
num_job = cfg.num_job
num_m = cfg.num_machine
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print('Current Device:', device)
rule_weight = {100: {"ATCS": [2.730, 1.153], "COVERT": 6.8},
               200: {"ATCS": [3.519, 1.252], "COVERT": 4.4},
               400: {"ATCS": [3.338, 1.209], "COVERT": 3.9}}

env = PMSP(num_job=num_job, num_m=num_m, reward_weight=[weight_tard, weight_setup],
           rule_weight=rule_weight[num_job])
agent = PPO(cfg, env.state_dim, env.action_dim, optimizer_name=optim, K_epoch=K_epoch).to(device)

modelpath = 'output/F4_no_division_greedy_lr_0.0001_K_1_T_1_2/5_5/model/episode-500.pt'
checkpoint = torch.load(modelpath)
agent.load_state_dict(checkpoint["model_state_dict"])
state = env.reset()
r_epi = 0.0
done = False
action_list = [0, 0, 0, 0]
mapping = {0: "SSPT", 1: "ATCS", 2: "MDD", 3: "COVERT"}
while not done:
    if env.n_route > 0:
        print('\tQueue:\t', [env.input_queue[i].feature for i in range(len(env.input_queue))])
        print('\t\t\t', [env.input_queue[i].is_tardy for i in range(len(env.input_queue))])
        print('\tCalling line:', env.calling_line.name, '\tSetting:\t', env.calling_setting)
        print('\t\tM0:\t', env.model['Machine 0'].setup)
        print('\t\tM1:\t', env.model['Machine 1'].setup)
        print('\t\tM2:\t', env.model['Machine 2'].setup)
        print('\t\tM3:\t', env.model['Machine 3'].setup)
        print('\t\tM4:\t', env.model['Machine 4'].setup)
        print('\tf1 : ', nround(state[:env.num_m], 4))
        print('\tf2 : ', nround(state[env.num_m:env.num_m + 4], 4))
        print('\tf3 : ', nround(state[env.num_m + 4:env.num_m + 8], 4))
        print('\tf4 : ', nround(state[env.num_m + 8:], 4))
    logit = agent.pi(torch.from_numpy(state).float().to(device))
    prob = torch.softmax(logit, dim=-1)
    """
    f_1 = np.zeros(self.num_m)  # Setup -> 현재 라인의 셋업 값과 같은 셋업인 job의 수
    f_2 = np.zeros(4)  # Due Date -> Tardiness level for non-setup
    f_3 = np.zeros(4)  # Due Date -> Tardiness level for setup
    f_4 = np.zeros(self.num_m)  # General Info -> 각 라인의 progress rate
    """
    action = torch.argmax(prob).item()
    next_state, reward, done = env.step(action)
    action_list[action] += 1

    # 남은 job이 20개 미만으로 떨어지면
    # if env.n_route > 60:
    #     print('\tNext Queue:\t', [env.input_queue[i].feature for i in range(len(env.input_queue))])
    #     if env.recent_tardy:
    #         print('\tThe agent picked a tardy job.')
        # print('\tRemaining tardy jobs...')
        # print('\tCalling line:', env.calling_line.name, '\tSetting:\t', env.calling_setting)
        # print('\t\tM0:\t', env.model['Machine 0'].setup)
        # print('\t\tM1:\t', env.model['Machine 1'].setup)
        # print('\t\tM2:\t', env.model['Machine 2'].setup)
        # print('\t\tM3:\t', env.model['Machine 3'].setup)
        # print('\t\tM4:\t', env.model['Machine 4'].setup)
        # print('\tf1 : ', nround(state[:env.num_m], 4))
        # print('\tf2 : ', nround(state[env.num_m:env.num_m + 4], 4))
        # print('\tf3 : ', nround(state[env.num_m + 4:env.num_m + 8], 4))
        # print('\tf4 : ', nround(state[env.num_m + 8:], 4))
    print(round(env.sim_env.now, 3), '\t|Probability:', tround(prob, 3))
    print(round(env.sim_env.now, 3), '\t| PPO chose ', mapping[action])
    print()
    state = next_state
    if done:
        tardiness = env.monitor.tardiness / env.num_job
        setup = env.monitor.setup / env.num_job
        makespan = env.sink.makespan
        break

print()
print(modelpath)
for i in range(4):
    print(mapping[i], ':\t', action_list[i])
print('Tardiness:', round(tardiness, 3))
print('setup:', round(setup, 3))
print('makespan:', round(makespan, 3))