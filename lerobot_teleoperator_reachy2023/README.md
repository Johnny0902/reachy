# lerobot_teleoperator_reachy2023

LeRobot teleoperator plugin for Reachy 2023.

- `--teleop.type=reachy2023_vr --teleop.mode=external` reads joint goals while the official Quest app moves the robot.
- `--teleop.type=reachy2023_vr --teleop.mode=quest` reads the custom Unity UDP stream and solves inverse kinematics.

Install `lerobot` and the main `reachy2023-pour` package first. See the repository README.
