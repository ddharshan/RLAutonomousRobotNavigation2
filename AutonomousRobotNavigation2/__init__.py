from gym.envs.registration import register
from AutonomousRobotNavigation2.environments import MovingEnv #change----------------------------------

register(
    id='Moving-v0',
    entry_point='AutonomousRobotNavigation2:MovingEnv',   #change-----------------------------------------
)
