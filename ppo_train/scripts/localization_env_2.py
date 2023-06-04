import gym
from gym import spaces
import time
import numpy as np

import math

import yaml
import os,glob
import random

# from stable_baselines3.common.env_util import make_vec_env
# from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv

# # Parallel environments
# vec_env = make_vec_env("CartPole-v1", n_envs=4)

# model = PPO("MlpPolicy", vec_env, verbose=1)
# model.learn(total_timesteps=25000)
# model.save("ppo_cartpole")

# del model # remove to demonstrate saving and loading

# model = PPO.load("ppo_cartpole")

# obs = vec_env.reset()
# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = vec_env.step(action)
#     vec_env.render("human")


class LocalizationEnv(gym.Env):
    def __init__(self, 
            path_ ,
            discrete_actions: bool = False):

        self.point_plicp     = []
        self.point_orb       = []
        self.point_res_orb   = []
        self.point_res_plicp = []
        self.point_real      = []

        self.diff_plicp      = []
        self.diff_orb        = []
        self.diff_res_orb    = []
        self.diff_res_plicp  = []
        self.diff_real       = []

        self.ori_plicp = []
        self.ori_orb   = []
        self.ori_real  = []
        # self.diff_ori_plicp = []
        # self.diff_ori_orb = []
        # self.diff_ori_real = []


        self.path = path_ + "/data"
        self.file_path = ''
        self.file_list = []
        self.file_amount = 0
        self.file_count = 0
        self.random_file = False

        self.log_reward        = 0.0
        self.log_weight_x      = 0.0
        self.log_weight_y      = 0.0
        self.log_weight_theta   = 0.0
        self.log_ORB_MSE_x     = 0.0
        self.log_ORB_MSE_y     = 0.0
        self.log_PLICP_MSE_x   = 0.0
        self.log_PLICP_MSE_y   = 0.0
        self.log_OUR_MSE_x     = 0.0
        self.log_OUR_MSE_y     = 0.0
        self.log_OUR_MSE_theta = 0.0


        # self.save_reward = []

        # if discrete_actions:
        #     self.action_space = spaces.MultiDiscrete(
        #         [xy_action_space, xy_action_space, rot_action_space])
        # else:
        #     self.action_space = spaces.Box(-1, 1, (3,))

        #  current_step define as current counter to which position, but this is count for diff, so need to minus one
        self.current_step = 0
        
        # self.action_space = spaces.MultiDiscrete(
        #         [100, 100])
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float64)

        self.GetYamlList()

        self.observation_space = spaces.Dict({
            'ORB_x'      : spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float64),
            'ORB_y'      : spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float64),
            'PLICP_x'    : spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float64),
            'PLICP_y'    : spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float64),
            'ORB_res_x'  : spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float64),
            'ORB_res_y'  : spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float64),
            'PLICP_res_x': spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float64),
            'PLICP_res_y': spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float64),
            'ORB_o'      : spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float64),
            'PLICP_o'    : spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float64)
        })

    def LoadParam(self):
        with open(self.file_path, 'r') as f:
            _data = yaml.safe_load(f)

        ######### save PLICP point ###########
        self.point_plicp.append([])
        self.point_plicp.append([])
        for i in range(0, int(len(_data['Trajectory']['PLICP'])/3)):
            self.point_plicp[0].append( _data['Trajectory']['PLICP'][3*i]/100 )
            self.point_plicp[1].append( _data['Trajectory']['PLICP'][3*i+1]/100 )

        ######### save ORB point ###########
        self.point_orb.append([])
        self.point_orb.append([])
        for i in range(0, int(len(_data['Trajectory']['ORB'])/3)):
            self.point_orb[0].append( _data['Trajectory']['ORB'][3*i]/100 )
            self.point_orb[1].append( _data['Trajectory']['ORB'][3*i+1]/100 )

        ######### save ORBSLAM2 Residual of x,y point  ###########
        self.point_res_orb.append([])
        self.point_res_orb.append([])
        for i in range(0, len(_data['Residual_ORB']['x'])):
            self.point_res_orb[0].append( _data['Residual_ORB']['x'][i]/100 )
            self.point_res_orb[1].append( _data['Residual_ORB']['y'][i]/100 )

        ######### save PLICP Residual of x,y point ###########
        self.point_res_plicp.append([])
        self.point_res_plicp.append([])
        for i in range(0, len(_data['Residual_PLICP']['x'])):
            self.point_res_plicp[0].append( _data['Residual_PLICP']['x'][i]/100 )
            self.point_res_plicp[1].append( _data['Residual_PLICP']['y'][i]/100 )

        ######### save real robot x,y point ###########
        self.point_real.append([])
        self.point_real.append([])
        for i in range(0, int(len(_data['Trajectory']['real'])/3)):
            self.point_real[0].append( _data['Trajectory']['real'][3*i]/100 )
            self.point_real[1].append( _data['Trajectory']['real'][3*i+1]/100 )

        ######## save plicp_rotate point ###########
        for i in range(0, int(len(_data['Rotate']['PLICP']))):
            self.ori_plicp.append(_data['Rotate']['PLICP'][i] )

        ######### save plicp_rotate point ###########
        for i in range(0, int(len(_data['Rotate']['ORB']))):
            self.ori_orb.append(_data['Rotate']['ORB'][i] )
            
        ######### save plicp_rotate point ###########
        for i in range(0, int(len(_data['Rotate']['real']))):
            self.ori_real.append(_data['Rotate']['real'][i] )


    def ComputeDiff(self):
        self.diff_plicp.append([])
        self.diff_plicp.append([])
        self.diff_plicp.append([])
        self.diff_orb.append([])
        self.diff_orb.append([])
        self.diff_orb.append([])
        self.diff_res_plicp.append([])
        self.diff_res_plicp.append([])
        self.diff_res_orb.append([])
        self.diff_res_orb.append([])
        self.diff_real.append([])
        self.diff_real.append([])
        self.diff_real.append([])

        for i in range (1, len(self.point_plicp[0])):
            self.diff_plicp[0].append(self.CheckLimit(self.point_plicp[0][i] - self.point_plicp[0][i-1]))
            self.diff_plicp[1].append(self.CheckLimit(self.point_plicp[1][i] - self.point_plicp[1][i-1]))
            self.diff_plicp[2].append(self.CheckOrientation(self.ori_plicp[i] - self.ori_plicp[i-1]))

            self.diff_orb[0].append(self.CheckLimit(self.point_orb[0][i] - self.point_orb[0][i-1]))
            self.diff_orb[1].append(self.CheckLimit(self.point_orb[1][i] - self.point_orb[1][i-1]))
            self.diff_orb[2].append(self.CheckOrientation(self.ori_orb[i] - self.ori_orb[i-1]))

            self.diff_res_plicp[0].append(self.CheckLimit(self.point_res_plicp[0][i] - self.point_res_plicp[0][i-1]))
            self.diff_res_plicp[1].append(self.CheckLimit(self.point_res_plicp[1][i] - self.point_res_plicp[1][i-1]))

            self.diff_res_orb[0].append(self.CheckLimit(self.point_res_orb[0][i] - self.point_res_orb[0][i-1]))
            self.diff_res_orb[1].append(self.CheckLimit(self.point_res_orb[1][i] - self.point_res_orb[1][i-1]))

            self.diff_real[0].append(self.CheckLimit(self.point_real[0][i] - self.point_real[0][i-1]))
            self.diff_real[1].append(self.CheckLimit(self.point_real[1][i] - self.point_real[1][i-1]))
            self.diff_real[2].append(self.CheckOrientation(self.ori_real[i] - self.ori_real[i-1]))

    def CheckLimit(self, diff_):
        if (diff_ > 1.0):
            return 1.0
        elif (diff_ < -1.0):
            return -1.0
        else:
            return diff_

    def CheckOrientation(self, diff_):
        if (diff_ <=  - 1.0 * math.pi):
            return (diff_ + 2.0 * math.pi)
        elif (diff_ >= math.pi):
            return (diff_ - 2.0 * math.pi)
        else:
            return diff_

    def GetYamlList(self):
        for filename in glob.glob(os.path.join(self.path, '*.yaml')):
            self.file_list.append(filename)
        self.file_amount = len(self.file_list)

    def reset(self):
        self.point_plicp     = []
        self.point_orb       = []
        self.point_res_orb   = []
        self.point_res_plicp = []
        self.point_real      = []

        self.diff_plicp      = []
        self.diff_orb        = []
        self.diff_res_orb    = []
        self.diff_res_plicp  = []
        self.diff_real       = []

        self.ori_plicp = []
        self.ori_orb   = []
        self.ori_real  = []
        # self.diff_ori_plicp = []
        # self.diff_ori_orb = []
        # self.diff_ori_real = []


        # if every yaml had read, use random to open yaml
        if (self.file_count >= self.file_amount):
            self.random_file = True

        if (self.random_file == True ):
            self.file_path = self.file_list[random.randint(0, self.file_amount-1)]
        else:
            # change to other yaml
            self.file_path = self.file_list[self.file_count]
            self.file_count += 1

        self.LoadParam()
        self.ComputeDiff()

        self.current_step = 0
        obs = self.UpdateObservation()
        # print("===========================================")
        # print("=============== one round =================")
        return obs

    def ComputeReward(self, plicp_x_w_, plicp_y_w_, plicp_w_theta_):
        reward = 0.0
        # print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
        # print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")

        ## count the current reward of localization change 
        for i in range(0, len(self.diff_real)):
            if (i == 0):
                plicp_w = plicp_x_w_
            elif (i == 1):
                plicp_w = plicp_y_w_
            elif (i == 2):
                plicp_w = plicp_w_theta_

            predict_ = self.diff_orb[i][self.current_step+1] * (1-plicp_w) + self.diff_plicp[i][self.current_step+1] * (plicp_w)

            ## situation 1:
            # real position is between orb and plicp
            if ((self.diff_real[i][self.current_step+1] <= self.diff_orb[i][self.current_step+1]    and \
                 self.diff_real[i][self.current_step+1] >= self.diff_plicp[i][self.current_step+1])  or \
                (self.diff_real[i][self.current_step+1] >= self.diff_orb[i][self.current_step+1]    and \
                 self.diff_real[i][self.current_step+1] <= self.diff_plicp[i][self.current_step+1])):

                orb_dis   = abs(self.diff_real[i][self.current_step+1] - self.diff_orb[i][self.current_step+1])
                plicp_dis = abs(self.diff_real[i][self.current_step+1] - self.diff_plicp[i][self.current_step+1])
                between_range = abs( self.diff_orb[i][self.current_step+1] - self.diff_plicp[i][self.current_step+1])

                if (orb_dis > plicp_dis):
                    shortest = plicp_dis
                    fartest = orb_dis
                else:
                    shortest = orb_dis
                    fartest = plicp_dis

                # if real position with both quarter of between_range, both side still in the between_range area
                if ( orb_dis >= between_range/4 and \
                     plicp_dis >= between_range/4):
                    if (abs(predict_ - self.diff_real[i][self.current_step+1]) <= between_range/4):
                        reward += 1 - (abs(predict_ - self.diff_real[i][self.current_step+1]) / (between_range/4))
                    else:
                        reward -= (abs(predict_ - self.diff_real[i][self.current_step+1]) - (between_range/4)) / (fartest - between_range/4)
                else :
                    if (abs(predict_ - self.diff_real[i][self.current_step+1]) <= (fartest - between_range/2)):
                        reward += 1 -(abs(predict_ - self.diff_real[i][self.current_step+1]) / (between_range/2 - shortest))
                    else:
                        reward -= ((abs(predict_ - self.diff_real[i][self.current_step+1]) - (between_range/2 - shortest) ) / (between_range/2))

            ## situation 2:
            # real position isn't between orb and plicp
            elif ((self.diff_real[i][self.current_step+1] > self.diff_orb[i][self.current_step+1]    and \
                   self.diff_real[i][self.current_step+1] > self.diff_plicp[i][self.current_step+1])  or \
                  (self.diff_real[i][self.current_step+1] < self.diff_orb[i][self.current_step+1]    and \
                   self.diff_real[i][self.current_step+1] < self.diff_plicp[i][self.current_step+1])):

                orb_dis   = abs(self.diff_real[i][self.current_step+1] - self.diff_orb[i][self.current_step+1])
                plicp_dis = abs(self.diff_real[i][self.current_step+1] - self.diff_plicp[i][self.current_step+1])
                between_range = abs( self.diff_orb[i][self.current_step+1] - self.diff_plicp[i][self.current_step+1])
                if (orb_dis > plicp_dis):
                    shortest = plicp_dis
                else:
                    shortest = orb_dis

                # if predict position is close to the algorithm which close to real position, give positive reward (+)
                # if predict position is close to the algorithm which far from real position, give nagitive reward (-)
                if ((abs(predict_ - self.diff_real[i][self.current_step+1]) - shortest) < (between_range/2)):
                    reward += ((abs(predict_ - self.diff_real[i][self.current_step+1]) - shortest) / (between_range/2))
                else:
                    reward -= ((abs(predict_ - self.diff_real[i][self.current_step+1]) - shortest - (between_range/2)) / (between_range/2))

        return reward

    def UpdateObservation(self):
        obs = { 'ORB_x'      : [self.diff_orb[0][self.current_step]],
                'ORB_y'      : [self.diff_orb[1][self.current_step]],
                'PLICP_x'    : [self.diff_plicp[0][self.current_step]],
                'PLICP_y'    : [self.diff_plicp[1][self.current_step]],
                'ORB_res_x'  : [self.diff_res_orb[0][self.current_step]],
                'ORB_res_y'  : [self.diff_res_orb[1][self.current_step]],
                'PLICP_res_x': [self.diff_res_plicp[0][self.current_step]],
                'PLICP_res_y': [self.diff_res_plicp[1][self.current_step]],
                'ORB_o'      : [self.diff_plicp[2][self.current_step]],
                'PLICP_o'    : [self.diff_orb[2][self.current_step]]}
        return obs

    def step(self, action):
        # action will return PLICP weight
        plicp_x_w = action[0]
        plicp_y_w = action[1]
        plicp_w_theta = action[2]

        reward = self.ComputeReward(plicp_x_w, plicp_y_w, plicp_w_theta)
        self.SaveLog(reward, action)

        self.current_step += 1

        # self.save_reward.append(reward)
        done = False
        if (len(self.diff_orb[0]) == self.current_step+1):
            done = True
            obs = None
        else:
            obs = self.UpdateObservation()

        info = {'info': 'hello'}
        return (obs, reward, done, info)

    def SaveLog(self, reward, action):
        self.log_reward       = reward
        self.log_weight_x     = action[0]
        self.log_weight_y     = action[1]
        self.log_weight_theta = action[2]
        self.log_ORB_MSE_x    = self.ComputeMSE(self.diff_real[0][self.current_step+1], self.diff_orb[0][self.current_step+1])
        self.log_ORB_MSE_y    = self.ComputeMSE(self.diff_real[1][self.current_step+1], self.diff_orb[1][self.current_step+1])
        self.log_PLICP_MSE_x  = self.ComputeMSE(self.diff_real[0][self.current_step+1], self.diff_plicp[0][self.current_step+1])
        self.log_PLICP_MSE_y  = self.ComputeMSE(self.diff_real[1][self.current_step+1], self.diff_plicp[1][self.current_step+1])

        our_x     = self.diff_orb[0][self.current_step+1] * (1-action[0]) + self.diff_plicp[0][self.current_step+1] * (action[0])
        our_y     = self.diff_orb[1][self.current_step+1] * (1-action[1]) + self.diff_plicp[1][self.current_step+1] * (action[1])
        our_theta = self.diff_orb[2][self.current_step+1] * (1-action[2]) + self.diff_plicp[2][self.current_step+1] * (action[2])

        self.log_OUR_MSE_x     = self.ComputeMSE(self.diff_real[0][self.current_step+1], our_x)
        self.log_OUR_MSE_y     = self.ComputeMSE(self.diff_real[1][self.current_step+1], our_y)
        self.log_OUR_MSE_theta = self.ComputeMSE(self.diff_real[2][self.current_step+1], our_theta)



    def ComputeMSE(self, real, method):
        return ((real - method) ** 2)


    def render(self, mode='human'):
        # 渲染环境状态（可选）
        pass



