import numpy as np
from typing import Tuple
from typing import Optional
from collections import namedtuple
from os import path

import gym
from gym import spaces
from gym.utils import seeding
gym.logger.set_level(40)  # noqa

from AutonomousRobotNavigation2.agents import BaseAgent  #change---------------------------------- 
from AutonomousRobotNavigation2.agents import MovingAgent  #change---------------------------------------



# Action Id
ACCELERATE = 0
TURN = 1
BREAK = 2


Target = namedtuple('Target', ['x', 'y', 'radius'])


class Action:
    """"
    Action class to store and standardize the action for the environment.
    """
    def __init__(self, id_: int, parameters: list):
        """"
        Initialization of an action.

        Args:
            id_: The id of the selected action.
            parameters: The parameters of an action.
        """
        self.id = id_
        self.parameters = parameters

    @property
    def parameter(self) -> float:
        """"
        Property method to return the parameter related to the action selected.

        Returns:
            The parameter related to this action_id
        """
        if len(self.parameters) == 2:
            return self.parameters[self.id]
        else:
            return self.parameters[0]


class BaseEnv(gym.Env):
    """"
    Gym environment parent class.
    """
    def __init__(
            self,
            seed: Optional[int] = None,
            max_turn: float = np.pi/9,
            max_acceleration: float = 1,
            delta_t: float = 0.005,
            max_step: int = 1000,
            penalty: float = 0.001,
            break_value: float = 0.1,
    ):
        """Initialization of the gym environment.

        Args:
            seed (int): Seed used to get reproducible results.
            max_turn (float): Maximum turn during one step (in radian).
            max_acceleration (float): Maximum acceleration during one step.
            delta_t (float): Time duration of one step.
            max_step (int): Maximum number of steps in one episode.
            penalty (float): Score penalty given at the agent every step.
            break_value (float): Break value when performing break action.
        """
        # Agent Parameters
        self.max_turn = max_turn
        self.max_acceleration = max_acceleration
        self.break_value = break_value

        # Environment Parameters
        self.delta_t = delta_t
        self.max_step = max_step
        self.field_size = 1.0
        self.target_radius = 0.05
        self.penalty = penalty

        # Initialization
        self.seed(seed)
        self.target = None
        self.viewer = None
        self.current_step = None
        self.agent = BaseAgent(break_value=break_value, delta_t=delta_t)
        self.pedestrian1 = BaseAgent(break_value=break_value, delta_t=delta_t)


        parameters_min = np.array([0, -1])
        parameters_max = np.array([1, +1])

        self.action_space = spaces.Tuple((spaces.Discrete(3),
                                          spaces.Box(parameters_min, parameters_max)))
        self.observation_space = spaces.Box(np.ones(20), -np.ones(20))  

    def seed(self, seed: Optional[int] = None) -> list:
        self.np_random, seed = seeding.np_random(seed)  # noqa
        return [seed]

    def reset(self) -> list:
        self.current_step = 0

        #self.target = Target(0.9,-0.85,self.target_radius) #position of the goal
        
        #self.agent.reset(-0.7, 0.6, np.pi, 0, 1, 1.5*(np.pi))  #Initiallizing the robot
        
        #Randomizing the initial position of pedestrian1 which will ensure the dynmaic environment(Rather than fixing the starting position of pedestrian, randomizing will be more effective. Because the dynamic is high)   
        low = [-1,-1, 0,-1, -1, 0]   #limiting the starting of randomization of pedestrian 1 greater than -0.3 in x plane which will ensure not collide with the robot at starting 
        high = [1,1,2*(np.pi), 1, 1, 2*(np.pi)]  
        
        #self.pedestrian1.reset(*self.np_random.uniform(low, high)) #randomizing the starting of pedestrian 1
        self.agent.reset(*self.np_random.uniform(low, high))  #Initiallizing the robot
        self.pedestrian1.reset(*self.np_random.uniform(low, high)) #randomizing the starting of pedestrian 1
        
 
        limit = self.field_size-self.target_radius
        lower = [-limit, -limit, self.target_radius]
        higher = [limit, limit, self.target_radius]
        self.target = Target(*self.np_random.uniform(lower, higher))
    
        #Initial acceleration which means, the constant velocity due to the absense of the increament in def step()
        self.pedestrian1.p1accelerate(1.2) #Constant speed or initial speed
       
        
      
        

        return self.get_state()

    def step(self, raw_action: Tuple[int, list]) -> Tuple[list, float, bool, dict]:
        action = Action(*raw_action)
        last_distance = self.distance
        self.current_step += 1

        if action.id == TURN:
            rotation = self.max_turn * max(min(action.parameter, 1), -1)
            self.agent.turn(rotation)
            if self.pedestrian1.p1y <= -1 or self.pedestrian1.p1y >= 1 or self.pedestrian1.p1x <= -1 or self.pedestrian1.p1x >= 1 : #q
                    self.pedestrian1.p1turn(np.pi) #Turn by 180 degree which will ensure the pedestrian is not out of the environment 
                    self.pedestrian1.p1accelerate(0) #Constant speed 
               
            else:
                    self.pedestrian1.p1accelerate(0) #Constant speed 
                   
                  
        elif action.id == ACCELERATE:
            if self.agent.speed <= 1:  #acceleration is applied up to 2m/s (agent's maximum speed is 2m/s. Due to increment of max=0.1, the max.speed is 1.999~2m/s)
                    acceleration = self.max_acceleration * max(min(action.parameter, 1), 0) #Randomizing the acceleration - The maximum acceleration is by 0.1. However, 0.1*1=0.1 also can be achieved
                    self.agent.accelerate(acceleration)
                    if self.pedestrian1.p1y <= -1 or self.pedestrian1.p1y >= 1 or self.pedestrian1.p1x <= -1 or self.pedestrian1.p1x >= 1 : #q
                            self.pedestrian1.p1turn(np.pi) #Turn by 180 degree which will ensure the pedestrian is not out of the environment 
                            self.pedestrian1.p1accelerate(0) #Constant speed 
               
                    else:
                            self.pedestrian1.p1accelerate(0) #Constant speed 
                        
            else:
                    acceleration = 0
                    self.agent.accelerate(acceleration)
                    
                    if self.pedestrian1.p1y <= -1 or self.pedestrian1.p1y >= 1 or self.pedestrian1.p1x <= -1 or self.pedestrian1.p1x >= 1 : #q
                            self.pedestrian1.p1turn(np.pi) #Turn by 180 degree which will ensure the pedestrian is not out of the environment 
                            self.pedestrian1.p1accelerate(0) #Constant speed 
                    else:
                            self.pedestrian1.p1accelerate(0) #Constant speed 
                           
                    
                    
        elif action.id == BREAK:
            self.agent.break_()
            if self.pedestrian1.p1y <= -1 or self.pedestrian1.p1y >= 1 or self.pedestrian1.p1x <= -1 or self.pedestrian1.p1x >= 1 : #q
                    self.pedestrian1.p1turn(np.pi) #Turn by 180 degree which will ensure the pedestrian is not out of the environment 
                    self.pedestrian1.p1accelerate(0) #Constant speed 
               
            else:
                    self.pedestrian1.p1accelerate(0) #Constant speed 
                   
                  
        
#------------------Expert Knowledge (Reward-function) -------------------------------------------


        if self.distance < self.target_radius and self.agent.speed == 0: #Robot reached the goal
            reward = self.get_reward(last_distance, True, False)
            done = True
        
        elif abs(self.agent.x) > self.field_size or abs(self.agent.y) > self.field_size or self.current_step > self.max_step: #Robot moving out of env and max.steps
            reward = -1
            done = True
            
        elif self.collision1 <= 0.21:  #Collision
            reward = -1 #changed
            done = True #changed
            
        elif (self.distance > 0.4) and (0.21 < self.collision1 < 0.35): #social-norm inducing reward for P1        
            
            if ((0.75*(np.pi)) < abs(self.thetaP1n - self.thetaRn) < np.pi):  #Passing of P1
                reward = self.get_reward(last_distance, False, True)
                done = False
            
            elif (((np.pi/4) < (self.thetaP1n - self.thetaRn) < 0.75*(np.pi)) and (abs(self.pedestrian1.p1speed) - abs(self.agent.speed) > 0)):  #Crossing of P1
                reward = self.get_reward(last_distance, False, True)
                done = False
            
            elif ((0 < (self.thetaRn - self.thetaP1n) < (np.pi/4)) and (abs(self.agent.speed) > abs(self.pedestrian1.p1speed))): #overtaking of P1
                reward = self.get_reward(last_distance, False, True)
                done = False
            
            else:
                reward = self.get_reward(last_distance) 
                done = False
        
        else:
            reward = self.get_reward(last_distance) 
            done = False
            
#------------------------------------------------------------------------------

        return self.get_state(), reward, done, {}

    def get_state(self) -> list:
        state = [
            
            #for robot and goal
            self.agent.x,
            self.agent.y,
            self.agent.speed,
            np.cos(self.agent.theta),
            np.sin(self.agent.theta),
            self.target.x,
            self.target.y,
            self.distance,
            0 if self.distance > self.target_radius else 1,
            self.current_step / self.max_step,
            
            #for collision1
            self.collision1,
            np.cos(self.pedestrian1.p1theta),
            np.sin(self.pedestrian1.p1theta),
            self.pedestrian1.p1x,
            self.pedestrian1.p1y,
            self.pedestrian1.p1speed,
            
            #pedestrian norm
            self.thetaP1n, 
            self.thetaRn, 
            self.agent.theta, #ask sir
            self.pedestrian1.p1theta, #ask sir
         
        ]
        return state

    def get_reward(self, last_distance: float, goal: bool = False, norm: bool = False) -> float: #changed
        return last_distance - self.distance - self.penalty + (1 if goal else 0)  + (0.0001*(1 if norm else 0))  #changed

#----------------------Define the distance-----------------

    @property
    def distance(self) -> float:
        return self.get_distance(self.agent.x, self.agent.y, self.target.x, self.target.y)

    @staticmethod
    def get_distance(x1: float, y1: float, x2: float, y2: float) -> float:
        return np.sqrt(((x1 - x2) ** 2) + ((y1 - y2) ** 2))

#----------------------Define the collision1---------------------
    @property
    def collision1(self) -> float:  # Define variables for difinition of collision
        return self.get_collision1(self.agent.x, self.agent.y, self.pedestrian1.p1x, self.pedestrian1.p1y)
    
    
    @staticmethod
    def get_collision1(x1: float, y1: float, x3: float, y3: float) -> float:   #Define the collision
        return np.sqrt(((x1 - x3) ** 2) + ((y1 - y3) ** 2))

    
#-------------------Define the thetaRn for norm to wrapp [-pi, pi]----------------------
    
    @property
    def thetaRn(self) -> float:  # Define variables for difinition of thetaRn
        return self.get_thetaRn(self.agent.theta)
    
    @staticmethod
    def get_thetaRn(x5: float) -> float:   #Define the thetaRn
        if x5 > np.pi:
            x5 = -(2*(np.pi)) + x5
        else:
            x5
        return x5

#-------------------Define the thetaP1n for norm to wrapp [-pi, pi]----------------------
    
    @property
    def thetaP1n(self) -> float:  # Define variables for difinition of thetaP1n
        return self.get_thetaP1n(self.pedestrian1.p1theta)
    
    @staticmethod
    def get_thetaP1n(x6: float) -> float:   #Define the thetaP1n
        if x6 > np.pi:
            x6 = -(2*(np.pi)) + x6
        else:
            x6
        return x6

    
#------------------------------------------------------------------




    def render(self, mode='human'):
        screen_width = 680
        screen_height = 680
        unit_x = screen_width / 2
        unit_y = screen_height / 2
        agent_inner_radius = 0.03
        agent_outer_radius = 0.04
        pedestrian1_radius = 0.0175


        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            
            #goal
            target = rendering.make_circle(unit_x * self.target_radius)
            target_trans = rendering.Transform(translation=(unit_x * (1 + self.target.x), unit_y * (1 + self.target.y)))
            target.add_attr(target_trans)
            target.set_color(0.80, 0, 0)
            self.viewer.add_geom(target)
            
            
            #Robot inner radius -1
            agentinner = rendering.make_circle(unit_x * agent_inner_radius)
            self.agentinner_trans = rendering.Transform(translation=(unit_x * (1 + self.agent.x), unit_y * (1 + self.agent.y)))  # noqa
            agentinner.add_attr(self.agentinner_trans)
            
            
            #Robot outer radius
            agentouter = rendering.make_circle(unit_x * agent_outer_radius)
            self.agentouter_trans = rendering.Transform(translation=(unit_x * (1 + self.agent.x), unit_y * (1 + self.agent.y)))  # noqa
            agentouter.add_attr(self.agentouter_trans)
            agentouter.set_color(1, 0.5, 0)
            self.viewer.add_geom(agentouter)
            
            #Robot inner radius- 2
            agentinner.set_color(0, 0, 0)
            self.viewer.add_geom(agentinner)
            
            #Robot's arrow
            t, r, m = 0.04 * unit_x, 0.0212 * unit_y, 0.0212 * unit_x
            arrow = rendering.FilledPolygon([(t, 0), (m, r), (m, -r)])
            self.arrow_trans = rendering.Transform(rotation=self.agent.theta)  # noqa
            arrow.add_attr(self.arrow_trans)
            arrow.add_attr(self.agentinner_trans)
            arrow.set_color(0, 0, 0)
            self.viewer.add_geom(arrow)

            
            
            #pedestrian 1
            
            #Head 1
            pedestrian1 = rendering.make_circle(unit_x * pedestrian1_radius)
            self.pedestrian1_trans = rendering.Transform(translation=(unit_x * (1 + self.pedestrian1.p1x), unit_y * (1 + self.pedestrian1.p1y)))  # noqa
            pedestrian1.add_attr(self.pedestrian1_trans)
            
            
            #P1Shoulder - right 
            p1rshoulder = rendering.make_capsule(8.432,16.66) # (rectangle length, arc diameter)
            self.p1rshoulder_trans = rendering.Transform(rotation=((self.pedestrian1.p1theta)+(np.pi/2)))  # noqa
            p1rshoulder.add_attr(self.p1rshoulder_trans)
            p1rshoulder.add_attr(self.pedestrian1_trans)
            p1rshoulder.set_color(0.5,0.5,0.5)
            self.viewer.add_geom(p1rshoulder)
            
            #P1Shoulder - left 
            p1lshoulder = rendering.make_capsule(-8.432,16.66) # (rectangle length, arc diameter)
            self.p1lshoulder_trans = rendering.Transform(rotation=((self.pedestrian1.p1theta)+(np.pi/2))) # noqa
            p1lshoulder.add_attr(self.p1lshoulder_trans)
            p1lshoulder.add_attr(self.pedestrian1_trans)
            p1lshoulder.set_color(0.5,0.5,0.5)
            self.viewer.add_geom(p1lshoulder)
            
            #Head -2
            pedestrian1.set_color(0, 0, 0)
            self.viewer.add_geom(pedestrian1)
            
            
            
            #P1arrow - 1
            a, b, c = 0.1705 * unit_x,  0.036  * unit_y, 0.108146 * unit_x
            p1arrow = rendering.FilledPolygon([(a, 0), (c, b), (c, -b)])
            self.p1arrow_trans = rendering.Transform(rotation=self.pedestrian1.p1theta)  # noqa 
            p1arrow.add_attr(self.p1arrow_trans)
            p1arrow.add_attr(self.pedestrian1_trans)
            
            
            
            
            #P1Stride 
            x, y, u = 0.0245 * unit_x, 0.036 * unit_y, 0.1705* unit_x
            p1stride = rendering.FilledPolygon([(x, y), (x, -y), (u, -y), (u, y)])
            self.p1stride_trans = rendering.Transform(rotation=self.pedestrian1.p1theta)  # noqa 
            p1stride.add_attr(self.p1stride_trans)
            p1stride.add_attr(self.pedestrian1_trans)
            p1stride.set_color(0.39, 0.58, 0.93)
            self.viewer.add_geom(p1stride)
            
            #P1arrow - 2
            p1arrow.set_color(0, 0, 0)
            self.viewer.add_geom(p1arrow)
           
        #Robot
        self.arrow_trans.set_rotation(self.agent.theta)
        self.agentinner_trans.set_translation(unit_x * (1 + self.agent.x), unit_y * (1 + self.agent.y))
        self.agentouter_trans.set_translation(unit_x * (1 + self.agent.x), unit_y * (1 + self.agent.y))
        
        #pedestrian 1
        self.p1arrow_trans.set_rotation(self.pedestrian1.p1theta)
        self.pedestrian1_trans.set_translation(unit_x * (1 + self.pedestrian1.p1x), unit_y * (1 + self.pedestrian1.p1y))
        #stridelength
        self.p1stride_trans.set_rotation(self.pedestrian1.p1theta)
        #shoulder
        self.p1rshoulder_trans.set_rotation((self.pedestrian1.p1theta)+(np.pi/2))
        self.p1lshoulder_trans.set_rotation((self.pedestrian1.p1theta)+(np.pi/2))
        
       
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


class MovingEnv(BaseEnv):
    def __init__(
            self,
            seed: int = None,
            max_turn: float = np.pi/9,
            max_acceleration: float = 1,
            delta_t: float = 0.005,
            max_step: int = 1000,
            penalty: float = 0.001,
            break_value: float = 0.1,
    ):

        super(MovingEnv, self).__init__(
            seed=seed,
            max_turn=max_turn,
            max_acceleration=max_acceleration,
            delta_t=delta_t,
            max_step=max_step,
            penalty=penalty,
            break_value=break_value,
        )

        self.agent = MovingAgent(
            break_value=break_value,
            delta_t=delta_t,
        )
        
        self.pedestrian1 = MovingAgent(
            break_value=break_value,
            delta_t=delta_t,
        )
        
 
