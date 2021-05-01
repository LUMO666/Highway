from onpolicy.envs.highway.highway_env.vehicle.kinematics import Vehicle
from onpolicy.envs.highway.highway_env.road.road import Road, LaneIndex, Route
from onpolicy.envs.highway.highway_env.types import Vector
from onpolicy.envs.highway.highway_env.vehicle.controller import ControlledVehicle

from typing import List, Tuple, Union
import numpy as np

class DummyVehicle(Vehicle):
    """ 
    The vehicle does not change lane and speed.
    """
    
    def __init__(self,
                 road: Road,
                 position: Vector,
                 heading: float = 0,
                 speed: float = 0,
                 target_lane_index: LaneIndex = None,
                 target_speed: float = None,
                 route: Route = None):
        super().__init__(road, position, heading, speed)
        self.target_lane_index = target_lane_index or self.lane_index
        self.target_speed = target_speed or self.speed
        self.target_speed = 25
        self.speed = 25
        self.route = route
    
    def act(self, action: Union[dict, str] = None) -> None:
        """
        An vehicle does not change its speed & lane.
        """
        action = {}
        action["steering"] = 0
        action['acceleration'] = 0
        Vehicle.act(self,action)
