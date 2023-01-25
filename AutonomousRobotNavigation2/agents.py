import numpy as np


class BaseAgent:
    def __init__(self, break_value: float, delta_t: float):
        self.x = None
        self.y = None
        self.phi = None  # angle of the velocity vector
        self.theta = None  # direction of the agent
        self.speed = None
        self.delta_t = delta_t
        self.break_value = break_value
        
        #pedestrian 1
        self.p1x = None
        self.p1y = None
        self.p1theta = None  # direction of the agent
        self.p1speed = None
        
        #pedestrian 2
        self.p2x = None
        self.p2y = None
        self.p2theta = None  # direction of pedestrian 2
        self.p2speed = None
        
        #pedestrian 3
        self.p3x = None
        self.p3y = None
        self.p3theta = None  # direction of pedestrian 3
        self.p3speed = None

        #pedestrian 4
        self.p4x = None
        self.p4y = None
        self.p4theta = None  # direction of pedestrian 4
        self.p4speed = None
        
        #pedestrian 5
        self.p5x = None
        self.p5y = None
        self.p5theta = None  # direction of pedestrian 5
        self.p5speed = None
        
        #pedestrian 6
        self.p6x = None
        self.p6y = None
        self.p6theta = None  # direction of pedestrian 6
        self.p6speed = None

    def accelerate(self, value: float) -> None:
        raise NotImplementedError

    def break_(self) -> None:
        raise NotImplementedError

    def turn(self, value: float) -> None:
        raise NotImplementedError

#pedestrian 1
    def p1accelerate(self, value: float) -> None:
        raise NotImplementedError

    def p1turn(self, value: float) -> None:
        raise NotImplementedError
        
        
#pedestrian 2
    def p2accelerate(self, value: float) -> None:
        raise NotImplementedError

    def p2turn(self, value: float) -> None:
        raise NotImplementedError
    
#pedestrian 3
    def p3accelerate(self, value: float) -> None:
        raise NotImplementedError

    def p3turn(self, value: float) -> None:
        raise NotImplementedError
        
        
#pedestrian 4
    def p4accelerate(self, value: float) -> None:
        raise NotImplementedError

    def p4turn(self, value: float) -> None:
        raise NotImplementedError

      
#pedestrian 5
    def p5accelerate(self, value: float) -> None:
        raise NotImplementedError

    def p5turn(self, value: float) -> None:
        raise NotImplementedError
        
        
#pedestrian 6
    def p6accelerate(self, value: float) -> None:
        raise NotImplementedError

    def p6turn(self, value: float) -> None:
        raise NotImplementedError
        
    def reset(self, x: float, y: float, direction: float, p1x: float, p1y: float, p1direction: float, p2x: float, p2y: float, p2direction: float, p3x: float, p3y: float, p3direction: float, p4x: float, p4y: float, p4direction: float,  p5x: float, p5y: float, p5direction: float,  p6x: float, p6y: float, p6direction: float) -> None:
        self.x = x
        self.y = y
        self.speed = 0
        self.theta = direction
#pedestrian 1
        self.p1x = p1x
        self.p1y = p1y
        self.p1speed = 0
        self.p1theta = p1direction
#pedestrian 2
        self.p2x = p2x
        self.p2y = p2y
        self.p2speed = 0
        self.p2theta = p2direction
        
#pedestrian 3
        self.p3x = p3x
        self.p3y = p3y
        self.p3speed = 0
        self.p3theta = p3direction

#pedestrian 4
        self.p4x = p4x
        self.p4y = p4y
        self.p4speed = 0
        self.p4theta = p4direction
        
#pedestrian 5
        self.p5x = p5x
        self.p5y = p5y
        self.p5speed = 0
        self.p5theta = p5direction
        
#pedestrian 6
        self.p6x = p6x
        self.p6y = p6y
        self.p6speed = 0
        self.p6theta = p6direction
        
    def _step(self) -> None:
        angle = self.theta if self.phi is None else self.phi
        self.x += self.delta_t * self.speed * np.cos(angle)
        self.y += self.delta_t * self.speed * np.sin(angle)
#pedestrian 1
        p1angle = self.p1theta 
        self.p1x += self.delta_t * self.p1speed * np.cos(p1angle)
        self.p1y += self.delta_t * self.p1speed * np.sin(p1angle)
#pedestrian 2
        p2angle = self.p2theta 
        self.p2x += self.delta_t * self.p2speed * np.cos(p2angle)
        self.p2y += self.delta_t * self.p2speed * np.sin(p2angle)
#pedestrian 3
        p3angle = self.p3theta 
        self.p3x += self.delta_t * self.p3speed * np.cos(p3angle)
        self.p3y += self.delta_t * self.p3speed * np.sin(p3angle)
#pedestrian 4
        p4angle = self.p4theta 
        self.p4x += self.delta_t * self.p4speed * np.cos(p4angle)
        self.p4y += self.delta_t * self.p4speed * np.sin(p4angle)
#pedestrian 5
        p5angle = self.p5theta 
        self.p5x += self.delta_t * self.p5speed * np.cos(p5angle)
        self.p5y += self.delta_t * self.p5speed * np.sin(p5angle)
#pedestrian 6
        p6angle = self.p6theta 
        self.p6x += self.delta_t * self.p6speed * np.cos(p6angle)
        self.p6y += self.delta_t * self.p6speed * np.sin(p6angle)







class MovingAgent(BaseAgent):
    def __init__(self, break_value: float, delta_t: float):
        super(MovingAgent, self).__init__(break_value, delta_t)

    def accelerate(self, value: float) -> None:
        self.speed += value
        self._step()

    def break_(self) -> None:
        self.speed = 0 if self.speed < self.break_value else self.speed - self.break_value
        self._step()

    def turn(self, value: float) -> None:
        self.theta = (self.theta + value) % (2 * np.pi)
        self._step()
#Pedestrian 1

    def p1accelerate(self, value: float) -> None:
        self.p1speed += value
        self._step()

    

    def p1turn(self, value: float) -> None:
        self.p1theta = (self.p1theta + value) % (2 * np.pi)
        self._step()
        
#Pedestrian 2

    def p2accelerate(self, value: float) -> None:
        self.p2speed += value
        self._step()

    def p2turn(self, value: float) -> None:
        self.p2theta = (self.p2theta + value) % (2 * np.pi)
        self._step()

#Pedestrian 3

    def p3accelerate(self, value: float) -> None:
        self.p3speed += value
        self._step()

    def p3turn(self, value: float) -> None:
        self.p3theta = (self.p3theta + value) % (2 * np.pi)
        self._step()
        
        
#Pedestrian 4

    def p4accelerate(self, value: float) -> None:
        self.p4speed += value
        self._step()

    def p4turn(self, value: float) -> None:
        self.p4theta = (self.p4theta + value) % (2 * np.pi)
        self._step()
        
#Pedestrian 5

    def p5accelerate(self, value: float) -> None:
        self.p5speed += value
        self._step()

    def p5turn(self, value: float) -> None:
        self.p5theta = (self.p5theta + value) % (2 * np.pi)
        self._step()

#Pedestrian 6

    def p6accelerate(self, value: float) -> None:
        self.p6speed += value
        self._step()

    def p6turn(self, value: float) -> None:
        self.p6theta = (self.p6theta + value) % (2 * np.pi)
        self._step()

 
