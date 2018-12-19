FROM xf1280/gibson:0.3.1

RUN apt install nano
RUN python3 -m pip install pyquaternion stable_baselines
