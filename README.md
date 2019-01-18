# OpenAI Gym environments for HAMSTIR Autonomous Mobile System for Testing Intelligent Robotics (HAMSTIR)

The goal of this project is to create a simple robot guided by a monocular camera
that can be trained end-to-end in simulation using reinforcement learning. This 
project is in early development, so not everything works yet ;) 

We lean heavily on domain randomization, with three rooms with random texture walls.
See a demo of a trained policy in one room (birds eye view on left, robot-eye view
  on right):

![PPO2 Policy Demo](https://github.com/abefetterman/hamstir-gym/raw/master/images/train_demo.gif "PPO2 Policy Demo")

## Dependencies

Compared to other robotic simulations intended for sim-to-real transfer, the
dependencies are light:

*  OpenAI `gym`
*  `pybullet >= 2.4.0`
*  `pyquaternion`

The `pybullet` environment makes use of texture and camera randomization to allow
sim-to-real transfer. Whether this is successful is yet to be shown.

For the Gibson testing environment, of course, `GibsonEnv` must be installed with 
its dependencies. The best way to do this is with Docker, and this project includes
a customized docker configuration that will include dependencies for this project
as well as the original `GibsonEnv`.

##  Environments

The training environment is `HamstirRoomEmptyEnv`, which is a 
[pybullet](https://github.com/bulletphysics/bullet3) simulation in a room with 
no objects. 

There is an additional testing environment, `HamstirGibsonEnv`, which uses the 
[Gibson](https://github.com/StanfordVL/GibsonEnv) environment. Currently, the simulation
seems to do pybullet-to-real transfer better than pybullet-to-gibson transfer, but
this is an area of active development.

## Getting started

To install, simply run:

```
git clone https://github.com/abefetterman/hamstir-gym
cd hamstir-gym
pip install -e .
```

Start training with:

```
python3 ./examples/train.py
``` 

Then run the policy with:

```
python3 ./examples/run.py --model /tmp/gym/best_model.pkl
```

## HamstirRoomEmptyEnv

The reward in this environment is based on forward motion distance, with a penalty
for wall collisions and wall proximity. There are three rooms, chosen randomly for
each episode:

![room6x6](https://github.com/abefetterman/hamstir-gym/raw/master/images/room6x6.png "room6x6") ![room2x12](https://github.com/abefetterman/hamstir-gym/raw/master/images/room2x12.png "room2x12") ![room12x12](https://github.com/abefetterman/hamstir-gym/raw/master/images/room12x12.png "room12x12")

## References
 
_Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World._ 
J Tobin, R Fong, A Ray, et. al. 
[arXiv:1703.06907](https://arxiv.org/abs/1703.06907) (2017).

_CAD2RL: Real Single-Image Flight without a Single Real Image._ 
F Sadeghi, S Levine.
[arXiv:1611.04201](https://arxiv.org/abs/1611.04201) (2016).



