""" train_ppo_cpu.py 로 학습한 모델을 불러와서 test하고 로그를 뽑는 코드 """

import os
import torch
import json
import math

# from cfg import get_cfg

from cfg_local import Configure
from agent.ppo import *
# from environment.env_jiwon import PMSP
from environment.env import PMSP

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)

if __name__ == "__main__":
    cfg = Configure()
    rule_weight = {100: {"ATCS": [2.730, 1.153], "COVERT": 6.8},
                   200: {"ATCS": [3.519, 1.252], "COVERT": 4.4},
                   400: {"ATCS": [3.338, 1.209], "COVERT": 3.9}}
    load_model = True
    weight = 0.6
    weight_tard = weight
    weight_setup = 1 - weight
    optim = cfg.optim
    learning_rate = cfg.lr
    K_epoch = cfg.K_epoch
    T_horizon = cfg.T_horizon
    num_episode = cfg.n_episode

    num_job = 100
    # num_job = cfg.num_job

    num_m = cfg.num_machine
    trained_model = "240620_10_lr_0.0001_K_1_T_1_2_5_5"
    keyword = "240621_test_ep600_" + str(num_job)
    dir = './output/{0}/'.format(keyword)
    if not os.path.exists(dir):
        os.makedirs(dir)

    with open(dir + "{0}_log.csv".format(keyword), 'w') as f:
        f.write('Type,Index,reward,r_tard,r_setup,SSPT,ATCS,MDD,COVERT,Avg_setup,Avg_tardiness,DDT,PT_var\n')

    mapping = {0: "SSPT", 1: "ATCS", 2: "MDD", 3: "COVERT", 4:"RL"}
    for i in [0, 1, 2, 3, 4]:
        for pt_var in [0.1, 0.2, 0.3, 0.4, 0.5]:
            for ddt in [0.8, 0.9, 1.0, 1.1, 1.2]:
                for idx in range(2):
                    env = PMSP(num_job=num_job, num_m=num_m, reward_weight=[weight_tard, weight_setup],
                               rule_weight=rule_weight[num_job], ddt=ddt, pt_var=pt_var, is_train=False)
                    if i == 4:
                        model_path = "C:\\SNU EnSite\\IPMSP\\output\\240620_10_lr_0.0001_K_1_T_1_2_5_5\\model\\episode-600.pt"
                        agent = PPO(cfg, env.state_dim, env.action_dim).to(device)
                        checkpoint = torch.load(model_path)
                        agent.load_state_dict(checkpoint["model_state_dict"])

                    state = env.reset()
                    r_epi = 0.0
                    done = False
                    action_list = [0, 0, 0, 0]
                    sum_tardy = 0  # 평균적으로 queue에 tardy했던 job의 수
                    while not done:
                        num_tardy = env.count_tardy()
                        sum_tardy += num_tardy
                        if i == 4:
                            logit = agent.pi(torch.from_numpy(state).float().to(device))
                            prob = torch.softmax(logit, dim=-1)
                            p = prob.tolist()
                            m = Categorical(probs=prob)  # 왜 에러가 떴을까,,,,,,
                            action = m.sample().item()
                        else:
                            action = i

                        action_list[action] += 1
                        next_state, reward, done = env.step(action)

                        if i == 4:
                            agent.put_data((state, action, reward, next_state, prob[action].item(), done))

                        state = next_state
                        r_epi += reward

                        if done:
                            tardiness = env.monitor.tardiness / env.num_job
                            setup = env.monitor.setup / env.num_job
                            mean_number_of_tardy_jobs = sum_tardy / env.num_job
                            makespan = env.sink.makespan
                            print("{3}\t{0}\tDDT {1}\tPT_var {2}\t|".format(idx,ddt,pt_var,mapping[i]),
                                  "reward: %.2f | Setup: %.2f | Tardiness %.2f | makespan %.2f | " %
                                  (r_epi, setup, tardiness, makespan),
                                  action_list[0], '\t', action_list[1], '\t', action_list[2], '\t', action_list[3])

                            with open(dir + "{0}_log.csv".format(keyword), 'a') as f:
                                f.write('%s,%d,%.2f,%.2f,%.2f,%d,%d,%d,%d,%.2f,%.2f,%.1f,%.1f\n' % (
                                    mapping[i],idx,r_epi, env.reward_tard, env.reward_setup, action_list[0], action_list[1],
                                    action_list[2], action_list[3], setup, tardiness, env.ddt, env.pt_var))
                            break
