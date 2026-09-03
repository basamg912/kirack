import torch
from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

from kirack.kirack_env import KirackEnvCfg
import gymnasium as gym

cfg = KirackEnvCfg()
cfg.scene.num_envs = 1

env = gym.make("Kirack-Velocity-Play-v0", cfg=cfg)
# obs, _ = env.reset()
# s = obs["policy"][0].cpu().numpy()

# print("total shape:", s.shape)
# print("ang_vel:", s[0:3])
# print("proj_gravity:", s[15:18])
# print("vel_cmd:", s[30:33])
# print("joint_pos_rel:", s[45:59])
# print("joint_vel_rel:", s[115:129])
# print("last_action:", s[185:216])
# print("torso_com first 10:", s[340:350])

# robot = env.unwrapped.scene["robot"]
# print("default joint pos:", robot.data.default_joint_pos[0].cpu().numpy())
# for i, name in enumerate(robot.joint_names):
#     lo = robot.data.soft_joint_pos_limits[0, i, 0].item()
#     hi = robot.data.soft_joint_pos_limits[0, i, 1].item()
#     print(f"[INFO] JOINT POS LIMIT : {name} lo : {lo} hi : {hi}")

robot = env.unwrapped.scene["robot"]
print(robot.joint_names)
print(robot.data.gear_ratio)


env.close()
simulation_app.close()
