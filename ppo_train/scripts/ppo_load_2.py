#!/usr/bin/env python3
from stable_baselines3 import PPO
from ppo_train.srv import PPOPose_
import rospy


class LoadPPO():
    def __init__(self, path_ ):
        rospy.init_node('load_ppo')
        self.ppo_model = PPO.load( path_)
        rospy.Service('/ppo/get_all_pose', PPOPose, self.ServicePPO)
        rospy.spin()

    def ServicePPO(self, req):
        obs = { 'ORB_x'      : [req.orb_x],
                'ORB_y'      : [req.orb_y],
                'PLICP_x'    : [req.plicp_x],
                'PLICP_y'    : [req.plicp_y],
                'ORB_res_x'  : [req.ukf_orb_x],
                'ORB_res_y'  : [req.ukf_orb_y],
                'PLICP_res_x': [req.ukf_plicp_x],
                'PLICP_res_y': [req.ukf_plicp_y]}

        action = self.ppo_model.predict(obs)

        dict = {'plicp_w_x': float(action[0][0]), 'plicp_w_y': float(action[0][1])}
        return dict

if __name__ == '__main__':
    path = rospy.get_param('/ppo_load/path_to_ppo')
    filename = "/ppo_localization_5m"

    startPPO = LoadPPO(path+filename)