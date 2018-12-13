from gibson.core.physics.robot_locomotors import WalkerBase
import numpy as np
import gym.spaces
import os
from hamstir_gym.utils import DATA_DIR

class Hamstirbot(WalkerBase):
    foot_list = []
    mjcf_scaling = 1
    model_type = "URDF"
    default_scale = 1
    
    def __init__(self, config, env=None):
        self.config = config
        scale = config["robot_scale"] if "robot_scale" in config.keys() else self.default_scale
        WalkerBase.__init__(self, "tb_base.urdf", "base_link", action_dim=4,
                            sensor_dim=20, power=2.5, scale=scale,
                            initial_pos=config['initial_pos'],
                            target_pos=config["target_pos"],
                            resolution=config["resolution"],
                            control = 'velocity',
                            env=env)
        self.physics_model_dir = DATA_DIR
        self.is_discrete = config["is_discrete"]

        if self.is_discrete:
            self.action_space = gym.spaces.Discrete(5)
            self.vel = .1
            self.action_list = [[self.vel, self.vel],
                                [-self.vel, -self.vel],
                                [self.vel, -self.vel],
                                [-self.vel, self.vel],
                                [0, 0]]

            self.setup_keys_to_action()
        else:
            action_high = 0.02 * np.ones([2])
            self.action_space = gym.spaces.Box(-action_high, action_high)

    def apply_action(self, action):
        if self.is_discrete:
            realaction = self.action_list[action]
        else:
            realaction = action
        WalkerBase.apply_action(self, realaction)

    def steering_cost(self, action):
        if not self.is_discrete:
            return 0
        if action == 2 or action == 3:
            return -0.1
        else:
            return 0

    def robot_specific_reset(self):
        WalkerBase.robot_specific_reset(self)

    def alive_bonus(self, z, pitch):
        return +1 # if z > 0.26 else -1  # 0.25 is central sphere rad, die if it scrapes the ground

    def setup_keys_to_action(self):
        self.keys_to_action = {
            (ord('w'),): 0,  ## forward
            (ord('s'),): 1,  ## backward
            (ord('d'),): 2,  ## turn right
            (ord('a'),): 3,  ## turn left
            (): 4
        }

    def calc_state(self):
        base_state = WalkerBase.calc_state(self)

        angular_speed = self.robot_body.angular_speed()
        return np.concatenate((base_state, np.array(angular_speed)))