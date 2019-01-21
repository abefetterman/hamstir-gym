### 21.01.2019

With the current models, after sim-to-real transfer the mobilenet model just 
goes straight (into the wall) and the nature-lite model just spins in circles. 
It looks like successful implementations in the past have used 
_collision prediction_ instead of _reward estimation_, so that is what I will
look into next. Will also discretize action choices to fwd-left, straight,
fwd-right, then we can just do classification on these three actions.

Mobilenet model is only retraining the top layer now. Should retrain all layers.
