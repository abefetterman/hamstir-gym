# OpenAI Gym environments for HAMSTIR Autonomous Mobile System for Testing Intelligent Robotics (HAMSTIR)

The goal of this project is to create a simple robot guided by a monocular camera
that can be trained end-to-end in simulation using reinforcement learning. This 
project is in early development, so not everything works yet ;) 

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

## Installing

Simply run:

```
git clone https://github.com/abefetterman/hamstir-gym
cd hamstir-gym
pip install -e .
```

 ## References
 
Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World. 
J Tobin, R Fong, A Ray, et. al. 
[arXiv:1703.06907](https://arxiv.org/abs/1703.06907) (2017).

CAD2RL: Real Single-Image Flight without a Single Real Image.
F Sadeghi, S Levine.
[arXiv:1611.04201](https://arxiv.org/abs/1611.04201) (2016).



