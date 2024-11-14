""" train_ppo_cpu.py 로 학습한 모델을 불러와서 test하고 로그를 뽑는 코드 """

"""
2024. 10. 08. environment에 log를 뽑기 위해 파일 분리
2024. 11. 14. RL 부분 모델 불러와서 테스트하는 코드 추가
"""
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

    with open("sample{0}.json".format(num_job), 'r') as f:
        sample_data = json.load(f)

    num_m = cfg.num_machine
    trained_model = "240620_10_lr_0.0001_K_1_T_1_2_5_5"
    # 폴더명으로 쓸 keyword 입력
    keyword = "241114_Heuristic_test_01_" + str(num_job)
    dirpath = './output/{0}/'.format(keyword)
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

    mapping = {0: "SSPT", 1: "ATCS", 2: "MDD", 3: "COVERT", 4:"RL"}
    pt_var = 0.2
    with open(dirpath + "overall.csv", 'w') as f:
        # column명 작성
        f.write('Heuristic,Test No.,PT_var,Makespan\n')
    with open(dirpath + "{0}_log.csv".format(keyword), 'w') as f:
        f.write('Method,Index,Reward,R_tardiness,R_setup,SSPT,ATCS,MDD,COVERT,Setup,Tardiness,DDT,PT_var\n')
    for i in [0, 1, 2, 3, 4]:
        for idx in range(1):
            test_data = sample_data[str(idx)]


            ddt = test_data['ddt']
            env = PMSP(num_job=num_job, test_sample=test_data,
                       rule_weight=rule_weight[num_job],
                       ddt=ddt, pt_var=pt_var, is_train=False)
            if i == 4:
                model_path = "C:\\SNU EnSite\\IPMSP\\output\\240620_12_lr_0.0001_K_1_T_1_2_5_5\\model\\episode-1.pt"
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

                # if i == 4:
                #     agent.put_data((state, action, reward, next_state, prob[action].item(), done))

                state = next_state
                r_epi += reward

                if done:
                    tardiness = env.monitor.tardiness / env.num_job
                    setup = env.monitor.setup / env.num_job
                    mean_number_of_tardy_jobs = sum_tardy / env.num_job
                    makespan = env.sink.makespan
                    env.monitor.get_logs(dirpath + "{0}_sample{1}_test{2}.csv".format(
                        mapping[i], num_job, str(idx)))
                    with open(dirpath + "overall.csv", 'a') as f:
                        # column명 작성
                        f.write('%s,%d,%.1f,%.2f\n'% (mapping[i], idx, pt_var, makespan))

                    print("{3}\t{0}\tDDT {1}\tPT_var {2}\t|".format(idx,ddt,pt_var,mapping[i]),
                          "reward: %.2f | Setup: %.2f | Tardiness %.2f | makespan %.2f | " %
                          (r_epi, setup, tardiness, makespan),
                          action_list[0], '\t', action_list[1], '\t', action_list[2], '\t', action_list[3])

                    with open(dirpath + "{0}_log.csv".format(keyword), 'a') as f:
                        f.write('%s,%d,%.2f,%.2f,%.2f,%d,%d,%d,%d,%.2f,%.2f,%.1f,%.1f\n' % (
                            mapping[i],idx,r_epi, env.reward_tard, env.reward_setup, action_list[0], action_list[1],
                            action_list[2], action_list[3], setup, tardiness, env.ddt, env.pt_var))
                        print(mapping[i]," File updated!")
                    break
