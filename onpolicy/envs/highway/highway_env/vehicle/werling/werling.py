from typing import List, Tuple, Union

import numpy as np
import copy
import math
import sys
sys.path.extend("../../")
from onpolicy.envs.highway.highway_env import utils
from onpolicy.envs.highway.highway_env.road.road import Road, LaneIndex, Route
from onpolicy.envs.highway.highway_env.types import Vector
from onpolicy.envs.highway.highway_env.vehicle.kinematics import Vehicle
from onpolicy.envs.highway.highway_env.vehicle.controller import ControlledVehicle
from onpolicy.envs.highway.highway_env.vehicle.werling.cubic_spline_planner import Spline2D
from onpolicy.envs.highway.highway_env.vehicle.werling.quintic_polynomials_planner import QuinticPolynomial

class QuarticPolynomial:

    def __init__(self, xs, vxs, axs, vxe, axe, time):
        # calc coefficient of quartic polynomial

        self.a0 = xs
        self.a1 = vxs
        self.a2 = axs / 2.0

        A = np.array([[3 * time ** 2, 4 * time ** 3],
                      [6 * time, 12 * time ** 2]])
        b = np.array([vxe - self.a1 - 2 * self.a2 * time,
                      axe - 2 * self.a2])
        x = np.linalg.solve(A, b)

        self.a3 = x[0]
        self.a4 = x[1]

    def calc_point(self, t):
        xt = self.a0 + self.a1 * t + self.a2 * t ** 2 + \
             self.a3 * t ** 3 + self.a4 * t ** 4

        return xt

    def calc_first_derivative(self, t):
        xt = self.a1 + 2 * self.a2 * t + \
             3 * self.a3 * t ** 2 + 4 * self.a4 * t ** 3

        return xt

    def calc_second_derivative(self, t):
        xt = 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t ** 2

        return xt

    def calc_third_derivative(self, t):
        xt = 6 * self.a3 + 24 * self.a4 * t

        return xt




class FrenetPath:

    def __init__(self):
        self.t = []
        self.d = []
        self.d_d = []
        self.d_dd = []
        self.d_ddd = []
        self.s = []
        self.s_d = []
        self.s_dd = []
        self.s_ddd = []
        self.cd = 0.0
        self.cv = 0.0
        self.cf = 0.0

        self.x = []
        self.y = []
        self.yaw = []
        self.ds = []
        self.c = []



class WerlingVehicle(ControlledVehicle):
    """ Parameter """
    SIM_LOOP = 500
    MAX_SPEED = 35.0  
    """ maximum speed [m/s] """
    MAX_ACCEL = 5.0  
    """ maximum acceleration [m/ss] """
    MAX_CURVATURE = 1.0  
    """ maximum curvature [1/m] """
    MIN_ROAD_Y_AXIS = 0.0  
    """ maximum road width [m] """
    MAX_ROAD_Y_AXIS = 16.0  
    """ maximum road width [m] """
    D_ROAD_W = 4.0  
    """ road width sampling length [m] """
    TARGET_SPEED = 30.0  
    """ target speed [m/s] """
    D_T_S = 5.0 
    """ target speed sampling length [m/s] """
    N_S_SAMPLE = 1  
    """ sampling number of target speed """
    LENGTH = 5.0     
    """ Vehicle length [m] """
    WIDTH = 2.5 
    """ Vehicle width [m] """

    """ parameters might change the performance"""
    DT = 0.2  #  DT = 1/simulation_frequency     
    """ time tick [s] """
    MAX_T = 3 
    """ max prediction time [m] """
    MIN_T = 2
    """ min prediction time [m] """
    SPEED_MAX = 30.0
    PERCEPTION_DISTANCE = 6.0 * SPEED_MAX 
    """ observed distance [m] """
    OBSERVED_VEHICLE_NUMBER = 5 
    """ observed number of vehicles [number] """
 
    """ cost weights """
    K_J = 0.1
    K_T = 0.1
    K_D = 1.0
    K_LAT = 1.0
    K_LON = 1.0

    def __init__(self,
                 road: Road,
                 position: Vector,
                 heading: float = 0,
                 speed: float = 25):
        super().__init__(road, position, heading, speed)
        self.wx = [0.0,  road.network.graph['0']['1'][0].length]
        self.wy = [0.0, 0.0]
        # initial state
        self.c_speed = speed  # current speed [m/s]
        self.c_d = position[1]  # current lateral position [m]
        self.c_d_d = 0.0  # current lateral speed [m/s]
        self.c_d_dd = 0.0  # current lateral acceleration [m/s]
        self.s0 = position[0]  # current course position
        self.tx, self.ty, self.tyaw, self.tc, self.csp = self.generate_target_course(self.wx , self.wy)


    @classmethod
    def make_on_lane(cls, road: Road, lane_index: LaneIndex, longitudinal: float, speed: float = 0) -> "WerlingVehicle":
        """
        Create a vehicle on a given lane at a longitudinal position.

        :param road: the road where the vehicle is driving
        :param lane_index: index of the lane where the vehicle is located
        :param longitudinal: longitudinal position along the lane
        :param speed: initial speed in [m/s]
        :return: A vehicle with at the specified position
        """
        lane = road.network.get_lane(lane_index)
        if speed is None:
            speed = lane.speed_limit
        return cls(road, lane.position(longitudinal, 0), lane.heading_at(longitudinal), speed)

    @classmethod
    def create_from(cls, vehicle: ControlledVehicle) -> "WerlingVehicle":
        """
        Create a new vehicle from an existing one.

        The vehicle dynamics and target dynamics are copied, other properties are default.

        :param vehicle: a vehicle
        :return: a new vehicle at the same dynamical state
        """
        v = cls(vehicle.road, vehicle.position, heading=vehicle.heading, speed=vehicle.speed,
                target_lane_index=vehicle.target_lane_index, target_speed=vehicle.target_speed,
                route=vehicle.route, timer=getattr(vehicle, 'timer', None))
        return v

    def act(self, action: Union[dict, str] = None):
        pass

    def step(self, dt: float):
        """
        Step the simulation.

        Increases a timer used for decision policies, and step the vehicle dynamics.

        :param dt: timestep
        """
        # self.timer += dt
        # super().step(dt)

        close_vehicles = self.road.close_vehicles_to(self, self.PERCEPTION_DISTANCE,
                                                         count=self.OBSERVED_VEHICLE_NUMBER-1,
                                                         see_behind=False)
        self.obstacle = np.array([[_vehicle.position[0], _vehicle.position[1], _vehicle.heading ]  for _vehicle in close_vehicles])
        # print(f"self.position = {self.position}")
        # print(f"self.obstacle = {self.obstacle}")
        # print(f"self.obstacle = {self.obstacle[:,0]}")
        
        path = self.frenet_optimal_planning(self.csp, self.s0, self.c_speed, self.c_d, self.c_d_d, self.c_d_dd, self.obstacle)
        # position, heading
        if path is None:
            return
        self.s0 = path.s[1]
        self.c_d = path.d[1]
        self.c_d_d = path.d_d[1]
        # print(f"lateral speed = {self.c_d_d}")
        self.c_d_dd = path.d_dd[1]
        self.c_speed = path.s_d[1]
        self.speed = np.linalg.norm([self.c_speed, self.c_d_dd])
        # print(f"self.speed = {self.speed}")
        # The DT Hyperparameter must be the same with the simulation time duration.
        self.position =  np.array([path.x[1], path.y[1]]).astype('float')
        self.heading = self.c_d_d / (self.LENGTH / 2) * dt
        self.on_state_update()


    def calc_frenet_paths(self, c_speed, c_d, c_d_d, c_d_dd, s0):
        frenet_paths = []

        # generate path to each offset goal
        for di in np.arange(self.MIN_ROAD_Y_AXIS, self.MAX_ROAD_Y_AXIS, self.D_ROAD_W):
            # Lateral motion planning
            for Ti in np.arange(self.MIN_T, self.MAX_T, self.DT):
                fp = FrenetPath()

                # lat_qp = quintic_polynomial(c_d, c_d_d, c_d_dd, di, 0.0, 0.0, Ti)
                lat_qp = QuinticPolynomial(c_d, c_d_d, c_d_dd, di, 0.0, 0.0, Ti)

                fp.t = [t for t in np.arange(0.0, Ti, self.DT)]
                fp.d = [lat_qp.calc_point(t) for t in fp.t]
                fp.d_d = [lat_qp.calc_first_derivative(t) for t in fp.t]
                fp.d_dd = [lat_qp.calc_second_derivative(t) for t in fp.t]
                fp.d_ddd = [lat_qp.calc_third_derivative(t) for t in fp.t]

                # Longitudinal motion planning (Velocity keeping)
                for tv in np.arange(self.TARGET_SPEED - self.D_T_S * self.N_S_SAMPLE,
                                    self.TARGET_SPEED + self.D_T_S * self.N_S_SAMPLE, self.D_T_S):
                    tfp = copy.deepcopy(fp)
                    lon_qp = QuarticPolynomial(s0, c_speed, 0.0, tv, 0.0, Ti)

                    tfp.s = [lon_qp.calc_point(t) for t in fp.t]
                    tfp.s_d = [lon_qp.calc_first_derivative(t) for t in fp.t]
                    tfp.s_dd = [lon_qp.calc_second_derivative(t) for t in fp.t]
                    tfp.s_ddd = [lon_qp.calc_third_derivative(t) for t in fp.t]

                    Jp = sum(np.power(tfp.d_ddd, 2))  # square of jerk
                    Js = sum(np.power(tfp.s_ddd, 2))  # square of jerk

                    # square of diff from target speed
                    ds = (self.TARGET_SPEED - tfp.s_d[-1]) ** 2

                    tfp.cd = self.K_J * Jp + self.K_T * Ti + self.K_D * tfp.d[-1] ** 2
                    tfp.cv = self.K_J * Js + self.K_T * Ti + self.K_D * ds
                    tfp.cf = self.K_LAT * tfp.cd + self.K_LON * tfp.cv

                    frenet_paths.append(tfp)

        return frenet_paths



    def calc_global_paths(self, fplist, csp):
        for fp in fplist:
            # import pdb
            # pdb.set_trace()
            # calc global positions
            for i in range(len(fp.s)):
                ix, iy = csp.calc_position(fp.s[i])
                
                if ix is None:
                    break
                i_yaw = csp.calc_yaw(fp.s[i])
                di = fp.d[i]
                fx = ix + di * math.cos(i_yaw + math.pi / 2.0)
                fy = iy + di * math.sin(i_yaw + math.pi / 2.0)
                fp.x.append(fx)
                fp.y.append(fy)

            # calc yaw and ds
            for i in range(len(fp.x) - 1):
                dx = fp.x[i + 1] - fp.x[i]
                dy = fp.y[i + 1] - fp.y[i]
                fp.yaw.append(math.atan2(dy, dx))
                fp.ds.append(math.hypot(dx, dy))

            fp.yaw.append(fp.yaw[-1])
            fp.ds.append(fp.ds[-1])

            # calc curvature
            for i in range(len(fp.yaw) - 1):
                fp.c.append((fp.yaw[i + 1] - fp.yaw[i]) / fp.ds[i])

        return fplist



    def point_in_rectangle(self, point: Vector, rect_min: Vector, rect_max: Vector) -> bool:
        """
        Check if a point is inside a rectangle

        :param point: a point (x, y)
        :param rect_min: x_min, y_min
        :param rect_max: x_max, y_max
        """
        return rect_min[0] <= point[0] <= rect_max[0] and rect_min[1] <= point[1] <= rect_max[1]



    def point_in_rotated_rectangle(self, point: np.ndarray, center: np.ndarray, length: float, width: float, angle: float) \
            -> bool:
        """
        Check if a point is inside a rotated rectangle

        :param point: a point
        :param center: rectangle center
        :param length: rectangle length
        :param width: rectangle width
        :param angle: rectangle angle [rad]
        :return: is the point inside the rectangle
        """
        c, s = np.cos(angle), np.sin(angle)
        r = np.array([[c, -s], [s, c]])
        ru = r.dot(point - center)
        return self.point_in_rectangle(ru, (-length/2, -width/2), (length/2, width/2))



    def has_corner_inside(self, 
                        rect1: Tuple[Vector, float, float, float],
                        rect2: Tuple[Vector, float, float, float]) -> bool:
        """
        Check if rect1 has a corner inside rect2

        :param rect1: (center, length, width, angle)
        :param rect2: (center, length, width, angle)
        """
        (c1, l1, w1, a1) = rect1
        (c2, l2, w2, a2) = rect2
        c1 = np.array(c1)
        l1v = np.array([l1/2, 0])
        w1v = np.array([0, w1/2])
        r1_points = np.array([[0, 0],
                            - l1v, l1v, -w1v, w1v,
                            - l1v - w1v, - l1v + w1v, + l1v - w1v, + l1v + w1v])
        c, s = np.cos(a1), np.sin(a1)
        r = np.array([[c, -s], [s, c]])
        rotated_r1_points = r.dot(r1_points.transpose()).transpose()
        return any([self.point_in_rotated_rectangle(c1+np.squeeze(p), c2, l2, w2, a2) for p in rotated_r1_points])



    def rotated_rectangles_intersect(self, 
                                    rect1: Tuple[Vector, float, float, float],
                                    rect2: Tuple[Vector, float, float, float]) -> bool:
        """
        Do two rotated rectangles intersect?

        :param rect1: (center, length, width, angle)
        :param rect2: (center, length, width, angle)
        :return: do they?
        """
        return self.has_corner_inside(rect1, rect2) or self.has_corner_inside(rect2, rect1)



    def check_collision_werling(self, fp):
        temp = False
        ob = self.obstacle
        if len(ob) == 0:
            return True

        for i in range(len(ob[:, 0])):
            for (ix, iy) in zip(fp.x, fp.y):
                if np.linalg.norm(np.array([ix, iy]) - np.array(ob[i,:2]) ) > self.LENGTH:
                    continue

                temp |= self.rotated_rectangles_intersect(([ix + self.LENGTH/2. , iy ], 0.9*self.LENGTH, 0.9*self.WIDTH, self.heading),
                                                        ([ob[i,0] + self.LENGTH/2. , ob[i,1]], 0.9*self.LENGTH, 0.9*self.WIDTH, ob[i,2]))
                
                # print(f"({ix:3.2f},{iy:3.2f}) ({ob[i, 0]}, {ob[i, 1]})collision?  {temp} ")
                if temp:
                    return False

            # d = [((ix - ob[i, 0]) ** 2 + (iy - ob[i, 1]) ** 2)
            #      for (ix, iy) in zip(fp.x, fp.y)]

            # collision = any([di <= ROBOT_RADIUS ** 2 for di in d])

            # if collision:
            #     return False

        return True


    def check_paths(self, fplist, ob):
        ok_ind = []
        for i, _ in enumerate(fplist):
            if any([v > self.MAX_SPEED for v in fplist[i].s_d]):  # Max speed check
                continue
            elif any([abs(a) > self.MAX_ACCEL for a in
                    fplist[i].s_dd]):  # Max accel check
                continue
            elif any([abs(c) > self.MAX_CURVATURE for c in
                    fplist[i].c]):  # Max curvature check
                continue
            elif not self.check_collision_werling(fplist[i]):
                continue

            ok_ind.append(i)

        return [fplist[i] for i in ok_ind]


    def frenet_optimal_planning(self, csp, s0, c_speed, c_d, c_d_d, c_d_dd, ob):
        fplist = self.calc_frenet_paths(c_speed, c_d, c_d_d, c_d_dd, s0)
        fplist = self.calc_global_paths(fplist, csp)
        fplist = self.check_paths(fplist, ob)
        # find minimum cost path
        min_cost = float("inf")
        best_path = None
        for fp in fplist:
            if min_cost >= fp.cf:
                min_cost = fp.cf
                best_path = fp

        return best_path


    def generate_target_course(self, x, y):
        csp = Spline2D(x, y)
        s = np.arange(0, csp.s[-1], 0.1)

        rx, ry, ryaw, rk = [], [], [], []
        # print("rx\try\tryaw\trk")
        for i_s in s:
            ix, iy = csp.calc_position(i_s)
            rx.append(ix)
            ry.append(iy)
            ryaw.append(csp.calc_yaw(i_s))
            rk.append(csp.calc_curvature(i_s))
            # print(f"{ix}\t{iy}\t{csp.calc_yaw(i_s)}\t{csp.calc_curvature(i_s)}")

        return rx, ry, ryaw, rk, csp


    def on_state_update(self) -> None:
        if self.road:
            self.lane_index = self.road.network.get_closest_lane_index(self.position, self.heading)
            self.lane = self.road.network.get_lane(self.lane_index)
            if self.road.record_history:
                self.history.appendleft(self.create_from(self))


