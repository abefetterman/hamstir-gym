from hamstir_gym.envs.hamstir_gibson_env import HamstirGibsonEnv
from gibson.utils.play import play
import argparse
import os

config_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'configs', 'gibson_train_allensville.yaml')
print(config_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=config_file)
    args = parser.parse_args()

    env = HamstirGibsonEnv(config = args.config)
    print(env.config)
    play(env, zoom=4)