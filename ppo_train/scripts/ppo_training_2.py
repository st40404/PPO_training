from stable_baselines3 import PPO
from localization_env_2 import LocalizationEnv
from datetime import datetime
from matplotlib import pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback

class TBCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # Log scalar value (here a random variable)
        self.logger.record("reward", self.locals['env'].get_attr('log_reward', 0)[0])
        self.logger.record("weight_x", self.locals['env'].get_attr('log_weight_x', 0)[0])
        self.logger.record("weight_y", self.locals['env'].get_attr('log_weight_y', 0)[0])
        self.logger.record("weight_theta", self.locals['env'].get_attr('log_weight_theta', 0)[0])
        self.logger.record("ORB_MSE_x", self.locals['env'].get_attr('log_ORB_MSE_x', 0)[0])
        self.logger.record("ORB_MSE_y", self.locals['env'].get_attr('log_ORB_MSE_y', 0)[0])
        self.logger.record("PLICP_MSE_x", self.locals['env'].get_attr('log_PLICP_MSE_x', 0)[0])
        self.logger.record("PLICP_MSE_y", self.locals['env'].get_attr('log_PLICP_MSE_y', 0)[0])
        self.logger.record("OUR_MSE_x", self.locals['env'].get_attr('log_OUR_MSE_x', 0)[0])
        self.logger.record("OUR_MSE_y", self.locals['env'].get_attr('log_OUR_MSE_y', 0)[0])
        self.logger.record("OUR_MSE_theta", self.locals['env'].get_attr('log_OUR_MSE_theta', 0)[0])

        return True

def main(path_):
    env = LocalizationEnv(path_)

    ppo_timesteps = 50000
    ppo_steps = 2048
    ppo_batch_size = 64
    ppo_rate = 0.0003


    ppo_params = {
        'n_steps': ppo_steps,  # 每次執行更新時的步數
        'batch_size': ppo_batch_size,  # 每次更新時的樣本批量大小
        'gamma': 0.99,  # 折扣因子
        'learning_rate': ppo_rate,  # 學習率
        'ent_coef': 0.01,  # 熵係數
        'clip_range': 0.2,  # Clip Range 參數
        'n_epochs': 10,  # 更新的迭代次數
        'gae_lambda': 0.95,  # GAE (Generalized Advantage Estimation) 的 lambda 參數
        'max_grad_norm': 0.5,  # 梯度裁剪的最大範圍
    }

    model = PPO("MultiInputPolicy", env, verbose=1, **ppo_params, tensorboard_log=path_ + '/log_2/')
    model.learn(total_timesteps=ppo_timesteps, tb_log_name= "{}_{}_{}_{}".format(ppo_timesteps, ppo_steps, ppo_batch_size, ppo_rate), callback=TBCallback())
    # model = PPO("MultiInputPolicy", env, verbose=1)
    # model.learn(total_timesteps=1000)
    model.save( path_ + "/result/ppo_localization")

    # ShowReward(env.save_reward, path_ + "/result/")


    del model # remove to demonstrate saving and loading

    print("========================   funish   ========================")
    print("========================   funish   ========================")
    print("========================   funish   ========================")


def ShowReward(reward, path):
    now = datetime.now()
    plt.plot(reward, color = 'red', linewidth ='1')
    plt.title( 'Reward')
    plt.xlabel('Step (s)')
    plt.ylabel('Reward')
    plt.savefig(path + now.strftime("%m%Y_%H:%M:%S"))
    plt.clf()

if __name__ == '__main__':
    path = "./src/ppo_train"
    main(path)
