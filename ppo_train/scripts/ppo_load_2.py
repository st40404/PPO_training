#!/usr/bin/env python3
from stable_baselines3 import PPO
from ppo_train.srv import PPOPose_2
import rospy


class LoadPPO():
    def __init__(self, path_ ):
        rospy.init_node('load_ppo_')
        self.ppo_model = PPO.load( path_)
        rospy.Service('/ppo/get_all_pose_', PPOPose_2, self.ServicePPO)
        rospy.spin()

    def ServicePPO(self, req):
        obs = { 'ORB_x'      : [req.orb_x],
                'ORB_y'      : [req.orb_y],
                'ORB_o'      : [req.orb_theta],
                'PLICP_x'    : [req.plicp_x],
                'PLICP_y'    : [req.plicp_y],
                'PLICP_o'    : [req.plicp_theta],
                'ORB_res_x'  : [req.ukf_orb_x],
                'ORB_res_y'  : [req.ukf_orb_y],
                'PLICP_res_x': [req.ukf_plicp_x],
                'PLICP_res_y': [req.ukf_plicp_y]}


        action = self.ppo_model.predict(obs)

        dict = {'plicp_w_x': float(action[0][0]), 'plicp_w_y': float(action[0][1]), 'plicp_w_theta': float(action[0][2])}
        return dict

if __name__ == '__main__':
    path = rospy.get_param('/ppo_load_2/path_to_ppo_')
    # filename = "/ppo_localization_5m_m2"
    # filename = "/m3_20m_0.00003"
    filename = "/m3_50m_0.003(r3_1)"

    startPPO = LoadPPO(path+filename)