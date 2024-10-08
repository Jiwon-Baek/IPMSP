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
    # if cfg.use_vessl:
    #     vessl.init(organization="snu-eng-dgx", project="Final-General-PMSP", hp=cfg)

    rule_weight = {100: {"ATCS": [2.730, 1.153], "COVERT": 6.8},
                   200: {"ATCS": [3.519, 1.252], "COVERT": 4.4},
                   400: {"ATCS": [3.338, 1.209], "COVERT": 3.9}}

    # rule_weight = {100: {"ATCS": [2.7, 1.2], "COVERT": 6.8},
    #                200: {"ATCS": [3.5, 1.3], "COVERT": 4.4},
    #                400: {"ATCS": [3.3, 1.2], "COVERT": 3.9}}

    load_model = False
    weight_list = [0.5]
    # weight_list = [0.0, 0.25, 0.5, 0.75, 1.0]
    # weight_list = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.0]
    for weight in weight_list:
        weight_tard = weight
        weight_setup = 1 - weight
        optim = cfg.optim
        learning_rate = cfg.lr
        K_epoch = cfg.K_epoch
        T_horizon = cfg.T_horizon
        num_episode = cfg.n_episode
        num_job = cfg.num_job
        num_m = cfg.num_machine
        keyword = "240620_12"

        dir = './output/{5}_lr_{0}_K_{1}_T_{2}_2_{3}_{4}/'.format(learning_rate, K_epoch, T_horizon,
                                                                              round(10 * weight),
                                                                              10 - round(10 * weight), keyword)
        if not os.path.exists(dir):
            os.makedirs(dir)

        model_dir = dir + 'model/'
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        simulation_dir = dir + 'simulation/'
        if not os.path.exists(simulation_dir):
            os.makedirs(simulation_dir)

        log_dir = dir + 'log/'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        env = PMSP(num_job=num_job, num_m=num_m, reward_weight=[weight_tard, weight_setup],
                   rule_weight=rule_weight[num_job])
        agent = PPO(cfg, env.state_dim, env.action_dim, optimizer_name=optim, K_epoch=K_epoch).to(device)
        start_episode = 1

        n_episode = cfg.n_episode
        n_record = 100
        # with open(log_dir + "train_log.csv", 'w') as f:
        with open(log_dir + "train_log.csv", 'w') as f:
            f.write('episode, reward, reward_tard, reward_setup, SSPT, ATCS, MDD, COVERT,Avg_tardy,DDT,PT_var\n')
        # for episode in range(start_episode, start_episode + num_episode):

        for episode in range(1, n_episode + 1):
            if episode % n_record == 0 or episode == 1:
                with open(log_dir + "train_log_episode_{0}.csv".format(episode), 'a') as f:
                    f.write('P(SSTP), P(ATCS), P(MDD), P(COVERT), Action, ,SSPT,ATCS,MDD,COVERT,n_tardy,pick_tardy\n')

            state = env.reset()
            r_epi = 0.0
            done = False
            action_list = [0, 0, 0, 0]
            mapping = {0: "SSPT", 1: "ATCS", 2: "MDD", 3: "COVERT"}
            sum_tardy = 0  # 평균적으로 queue에 tardy했던 job의 수
            while not done:
                num_tardy = env.count_tardy()
                sum_tardy += num_tardy
                logit = agent.pi(torch.from_numpy(state).float().to(device))
                prob = torch.softmax(logit, dim=-1)
                p = prob.tolist()

                m = Categorical(probs=prob)  # 왜 에러가 떴을까,,,,,,
                # m = Categorical(logit)

                # action = torch.argmax(prob).item()
                action = m.sample().item()
                action_list[action] += 1
                next_state, reward, done = env.step(action)

                pick_tardy = True if env.recent_tardy[-1] else False
                if episode % n_record == 0 or episode == 1:
                    with open(log_dir + "train_log_episode_{0}.csv".format(episode), 'a') as f:
                        f.write('{0},{1},{2},{3},{4},,{5},{6},{7},{8},{9},{10}\n'.format(p[0], p[1], p[2], p[3],
                                                                                         mapping[action],
                                                                                         action_list[0], action_list[1],
                                                                                         action_list[2], action_list[3],
                                                                                         num_tardy, pick_tardy))

                agent.put_data((state, action, reward, next_state, prob[action].item(), done))
                state = next_state

                r_epi += reward
                if done:
                    tardiness = env.monitor.tardiness / env.num_job
                    setup = env.monitor.setup / env.num_job
                    mean_number_of_tardy_jobs = sum_tardy/env.num_job
                    makespan = env.sink.makespan
                    weight = round(weight, 1)
                    print("{0}_{1} |".format(int(10 * weight), int(round(10 * (1 - weight), 1))),
                          "episode: %d | reward: %.4f | Setup: %.4f | Tardiness %.4f | makespan %.4f | " % (
                              episode, r_epi, setup, tardiness, makespan),
                          action_list[0], '\t', action_list[1], '\t', action_list[2], '\t', action_list[3])
                    # with open(log_dir + "train_log.csv", 'a') as f:
                    with open(log_dir + "train_log.csv", 'a') as f:
                        f.write('%d,%.2f,%.2f,%.2f,%d,%d,%d,%d,%.2f,%.1f,%.1f\n' % (
                            episode, r_epi, env.reward_tard, env.reward_setup, action_list[0], action_list[1],
                            action_list[2], action_list[3],mean_number_of_tardy_jobs,env.ddt,env.pt_var))
                    break
            agent.train_net()

            # if episode % 100 == 0 or episode == 1:
            #     agent.save(episode, model_dir)
            if episode % n_record == 0 or episode == 1:
                agent.save(episode, model_dir)
