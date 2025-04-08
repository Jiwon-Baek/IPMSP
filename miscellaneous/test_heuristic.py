import os
import json
import math


from cfg_local import Configure
from agent.ppo import *
from agent.heuristic import Heuristic
# from environment.env_jiwon import PMSP
from environment.env import PMSP

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

if __name__ == "__main__":
    cfg = Configure(100,5)

    rule_weight = {100: {"ATCS": [2.730, 1.153], "COVERT": 6.8},
                   200: {"ATCS": [3.519, 1.252], "COVERT": 4.4},
                   400: {"ATCS": [3.338, 1.209], "COVERT": 3.9}}

    load_model = False
    # weight_list = [0.5]
    weight_list = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.0]
    dir = './output/Heuristics/'
    if not os.path.exists(dir):
        os.makedirs(dir)
    filename = "Heuristics"+str(cfg.num_job)+".csv"
    with open(dir + filename, 'w') as f:
        f.write('idx,Heuristic,w_tard,w_setup,reward,reward_tard,reward_setup,setup,tardiness \n')

    map = {0: "SSPT", 1: "ATCS", 2: "MDD", 3: "COVERT"}
    for idx in range(10):
        for action in range(4):
            for weight in weight_list:
                weight_tard = weight
                weight_setup = 1 - weight
                # optim = "Adam"
                # learning_rate = 0.005 if not cfg.use_vessl else cfg.lr
                # K_epoch = 1 if not cfg.use_vessl else cfg.K_epoch
                # T_horizon = "entire" if not cfg.use_vessl else cfg.T_horizon
                num_episode = 1
                num_job = cfg.num_job
                num_m = cfg.num_machine
                keyword = map[action]

                env = PMSP(num_job=num_job, num_m=num_m, reward_weight=[weight_tard, weight_setup], rule_weight=rule_weight[num_job])
                state = env.reset()
                r_epi = 0.0
                done = False

                while not done:

                    next_state, reward, done = env.step(action)

                    # agent.put_data((state, action, reward, next_state, prob[action].item(), done))
                    # state = next_state
                    r_epi += reward

                    if done:
                        tardiness = env.monitor.tardiness / env.num_job
                        setup = env.monitor.setup / env.num_job
                        makespan = env.sink.makespan
                        print("{0}_{1} |".format(int(10 * weight), int(10 * (1 - weight))),
                              "Heuristic: %s | reward: %.4f | Setup: %.4f | Tardiness %.4f | makespan %.4f" % (
                              map[action], r_epi, setup, tardiness, makespan))

                        break
                with open(dir + filename, 'a') as f:
                    f.write('%d,%s,%.1f,%.1f,%.2f,%.2f,%.2f,%.4f,%.4f\n' % (idx,map[action], weight, round((1 - weight),1), r_epi, env.reward_tard, env.reward_setup, setup, tardiness))