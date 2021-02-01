from stable_baselines.common.env_checker import check_env
from stable_baselines.a2c import A2C
from .dc_env import DataCenterEnv
env = DataCenterEnv(arg1, ...)
# It will check your custom environment and output additional warnings if needed
check_env(env)
model = A2C('CnnPolicy', env).learn(total_timesteps=1000)
