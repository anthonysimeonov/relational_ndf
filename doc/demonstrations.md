# Collecting robot demonstrations and encoding with R-NDF
```
cd src/rndf_robot/demonstrations
python relational_demos.py --exp test_bottle_in_container --parent_class syn_container --child_class bottle --is_child_shapenet_obj
```

## Notes on teleoperation for collecting demonstrations
The script at [`demonstrations/relational_demos.py`](../src/rndf_robot/demonstrations/relational_demos.py) runs a super simple teleoperation setup for moving around a simulated Panda robot in PyBullet. We use it to provide rearrangement demonstrations in our three tasks. To make simulated rearrangement easier to demonstration, we artificially constrain the objects to the gripper when a grasp command is executed, even if the objects are not being physically grasped.

The keys `Q|E|A|D|S|X` are used to move `down|up|left|right|forward|back`. The keys `Z|C` are used to `open|close` the gripper. The keys `U|O|I|K|J|L` are used to rotate about the `X|Y|Z` axes of the gripper. The `G` key is used to toggle between the object that moves when the gripper is closed. 

At each step, a random object will be sampled and initialized in the environment. The `2` key is used to save the initial and final state of the objects in the scene, along with their point cloud observations.
