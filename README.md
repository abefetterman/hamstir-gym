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

## Environment: `HamstirRoomEmptyEnv`

This environment is designed for training the model. It simulates the robots in 
rooms with no other objects. There are three rooms, chosen randomly for
each episode:

![room6x6](https://github.com/abefetterman/hamstir-gym/raw/master/images/room6x6.png "room6x6") ![room2x12](https://github.com/abefetterman/hamstir-gym/raw/master/images/room2x12.png "room2x12") ![room12x12](https://github.com/abefetterman/hamstir-gym/raw/master/images/room12x12.png "room12x12")

The robot is initialized in the same location (0,2) with a random orientation. Wall
textures and camera angles are randomized for each episode.

The reward in this environment is based on forward motion distance, with a penalty
for wall collisions and wall proximity.

## Environment: `HamstirGibsonEnv`

This environment uses the [Gibson](https://github.com/StanfordVL/GibsonEnv) simulation. 
Currently, the model seems to do pybullet-to-gibson transfer no better than 
pybullet-to-real transfer, but this is an area of active development.

For the Gibson testing environment, `GibsonEnv` must be installed with 
its dependencies. The best way to do this is with Docker, and this project includes
a customized docker configuration that will include dependencies for this project
as well as the original `GibsonEnv`. 

## Running inference on a robot

These models are designed to be run on the 
[AIY Vision Kit](https://aiyprojects.withgoogle.com/vision/). The frame rate for 
Mobilenet-V2 has been around 7 fps--pretty good throughput for this little guy! 
The process for running a model is as follows:

Assume you have your model saved in `./models/my_model.pkl`. Then check 
`./examples/export.py` and make sure the policy is the same one you trained with,
for example, `NatureLitePolicy`. Then run:

```
python3 ./examples/export.py --model ./models/my_model.pkl --graph_out ./models/graph.pb
```

The next step relies on a TensorFlow model compiler from Google that only works 
(in my experience) on x86 Linux, so on a server or in a virtualenv, download and extract 
[bonnet_model_compiler.par](https://dl.google.com/dl/aiyprojects/vision/bonnet_model_compiler_latest.tgz). [See Google's instructions here](https://aiyprojects.withgoogle.com/vision/#makers-guide).

Then you will need the input and output graph operation names. These can be found 
by using TensorBoard--you may need some fiddling to avoid forbidden operations 
like division. For `NatureLitePolicy`, run:

```
./bonnet_model_compiler.par --frozen_graph_path ./graph.pb --output_graph_path outgraph.bp --input_tensor_name input/truediv --output_tensor_names model/pi/add --input_tensor_size 192
```

Then follow instructions on [hamstir-driver](https://github.com/abefetterman/hamstir-driver)
to run inference on the RPi-Zero of the AIY kit. 

## Research notes

Ongoing research notes are available in [NOTES.md](./NOTES.md)

## References
 
_Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World._ 
J Tobin, R Fong, A Ray, et. al. 
[arXiv:1703.06907](https://arxiv.org/abs/1703.06907) (2017).

_CAD2RL: Real Single-Image Flight without a Single Real Image._ 
F Sadeghi, S Levine.
[arXiv:1611.04201](https://arxiv.org/abs/1611.04201) (2016).

_Self-supervised Deep Reinforcement Learning with Generalized Computation Graphs for Robot Navigation_
G Kahn, A Villaflor, B Ding, P Abbeel, S Levine
[arXiv:1709.10489](https://arxiv.org/abs/1709.10489) (2017).
