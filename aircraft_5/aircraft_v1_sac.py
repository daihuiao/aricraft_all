import copy
import math
import pickle

import matplotlib.pyplot as plt
import numpy as np
import csv, os

import torch
import gym
from gym.spaces import Discrete, Box
# import gymnasium as gym
# from gymnasium.spaces import Box
import wandb
# box = Box(0.0, 1.0, shape=(3, 4, 5))

import scipy.io as sc
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime
from gen_cube import  Cube_generator, One_cube


# import matplotlib.pyplot as plt
# plt.switch_backend('agg')
from gen_cube import One_cube
#######train路径图#######
figure_dir = r'./figure'
if not os.path.exists(figure_dir):
    os.mkdir(figure_dir)
######train路径点########
record_dir = r'./history'
if not os.path.exists(record_dir):
    os.mkdir(record_dir)
#######test路径点########
test_dir = r'./test'
if not os.path.exists(test_dir):
    os.mkdir(test_dir)
#######test路径图#######
test_figure = r'./test/figure'
if not os.path.exists(test_figure):
    os.mkdir(test_figure)
current_path = r'./current_data/'
ucurrent = 'Current_u.mat'
vcurrent = 'Current_v.mat'
wcurrent = 'Current_w.mat'
kcurrent = 'Current_k.mat'
hcurrent = 'h.mat'
########Environment setting##########
START_POINT = np.array([1., 1., 6.])
GOAL_POINT = np.array([49., 49., 6.])
MIN = 0.0
MAX = 49.0

########Obs##########
obs1 = [40.0, 12.0, 35.0];
obs2 = [6.0, 35.0, 46.0];
obs3 = [35.0, 28.0, 20.0];
obs4 = [16.0, 30.0, 36.0];
obs5 = [4.0, 7.0, 30.0];
obs6 = [36.0, 39.0, 45.0];
obs7 = [15.0, 10.0, 26.0];
obs8 = [28.0, 40.0, 37.0];
obs9 = [20.0, 30.0, 30.0];
obs10 = [46.0, 18.0, 27.0];
obs11 = [47.0, 6.0, 12.0];
obs12 = [33.0, 19.0, 18.0];
obs13 = [6.0, 48.0, 47.0];
obs14 = [35.0, 36.0, 19.0];
obs15 = [29.0, 12.0, 45.0];
obs16 = [25.0, 20.0, 19.0];
Obs = [obs1, obs2, obs3, obs4, obs5, obs6, obs7, obs8, obs9, obs10, obs11, obs12, obs13, obs14, obs15, obs16]


class Obs():
    def __init__(self, position, static=True):
        self.position = position
        self.static = static

    def is_collision(self, AUV_position):
        if np.sqrt(np.sum((self.position - AUV_position) ** 2)) <= 3:
            return True
        else:
            return False


class Static_obs():
    def __init__(self, position, static=True):
        self.position = position
        self.static = static

    def is_collision(self, AUV_position):
        if np.sqrt(np.sum((self.position - AUV_position) ** 2)) <= 3:
            return True
        else:
            return False


class Dynnmic_obs():
    def __init__(self, position, static=False):
        self.position = position
        self.static = static

    def step_move(self):
        direction = np.random.randint(6)
        if direction == 0:  # x+
            self.position[0] += 1
        elif direction == 1:  # x-
            self.position[0] -= 1
        elif direction == 2:  # y+
            self.position[1] += 1
        elif direction == 3:  # y-
            self.position[1] -= 1
        elif direction == 4:  # z+
            self.position[2] += 1
        else:  # z-
            self.position[2] -= 1
        self.is_position_valid()

    def is_position_valid(self):
        for i in range(3):
            if self.position[i] > MAX:
                self.position[i] = MAX
            if self.position[i] < MIN:
                self.position[i] = MIN

    def is_collision(self, AUV_position):
        self.step_move()
        if np.sqrt(np.sum((self.position - AUV_position) ** 2)) <= 3:
            return True
        else:
            return False


# from utils_dai import plot_cuboid
# def mPrint():
#     center = [6, 6, 6]
#     length = 10
#     width = 10
#     height = 10
#     plot_cuboid(center, (length, width, height))
#
# mPrint()



# class One_cube():
#     def __init__(self, x, y, dx=None, dy=None, dz=None):  # 这里的x,y是中心点的坐标,并且中心都是1.5 3.5 等
#         self.x = x
#         self.y = y
#         if dx is None and dy is None and dz is None:
#             self.dx = (np.random.rand() + 0.2) / 1.2
#             self.dy = (np.random.rand() + 0.2) / 1.2
#             # self.dz = 2 * (np.random.rand() + 0.2) / 1.2 # 这里设置为2 让他更像一栋楼，但是
#             self.dz = (np.random.rand() + 0.2) / 1.2
#         else:
#             self.dx = dx
#             self.dy = dy
#             self.dz = dz


# class Cube_generator():
#     def __init__(self, grid_x=10, grid_y=10):
#         self.grid_x = grid_x
#         self.grid_y = grid_y
#
#         matrix = []
#         for i in range(grid_x):
#             row = []
#             for j in range(grid_y):
#                 row.append(0)  # 添加初始值，这里使用0
#             matrix.append(row)
#         self.cube = matrix
#
#         for y in range(self.grid_y):
#             for x in range(self.grid_x):
#                 one_cube = One_cube(x + 0.5, y + 0.5)
#                 self.check_valid(one_cube, x, y)
#                 self.cube[x][y] = one_cube
#         self.cube_num = 0
#
#     def check_valid(self, onecube, x, y):
#         if x > 0:
#             assert abs(onecube.x - self.cube[x - 1][y].x) > 0.5 * abs(onecube.dx + self.cube[x - 1][y].dx)
#         if y > 0:
#             assert abs(onecube.y - self.cube[x][y - 1].y) > 0.5 * abs(onecube.dy + self.cube[x][y - 1].dy)
#
#     def get_obs_util(self, x, y, flag, x_blank=None, y_blank=None):
#         x, y = math.floor(x), math.floor(y)
#         if flag == 0:
#             # 左上
#             x0, y0 = x - 1, y - 1
#             x1, y1 = x, y - 1
#             x2, y2 = x - 1, y
#             x3, y3 = x, y
#         elif flag == 1:
#             # 左下
#             x0, y0 = x - 1, y
#             x1, y1 = x, y
#             x2, y2 = x - 1, y + 1
#             x3, y3 = x, y + 1
#         elif flag == 2:
#             # 右上
#             x0, y0 = x, y - 1
#             x1, y1 = x + 1, y - 1
#             x2, y2 = x, y
#             x3, y3 = x + 1, y
#         elif flag == 3:
#             # 右下
#             x0, y0 = x, y
#             x1, y1 = x + 1, y
#             x2, y2 = x, y + 1
#             x3, y3 = x + 1, y + 1
#         else:
#             raise ValueError("flag error")
#
#         try:
#             cube_observation0 = [self.cube[x0][y0].x, self.cube[x0][y0].y, self.cube[x0][y0].dx,
#                                  self.cube[x0][y0].dy, self.cube[x0][y0].dz, ]
#         except:
#             pause = True
#
#         try:
#             cube_observation1 = [self.cube[x1][y1].x, self.cube[x1][y1].y, self.cube[x1][y1].dx,
#                                  self.cube[x1][y1].dy, self.cube[x1][y1].dz, ]
#         except:
#             pause = True
#
#         try:
#             cube_observation2 = [self.cube[x2][y2].x, self.cube[x2][y2].y, self.cube[x2][y2].dx,
#                                  self.cube[x2][y2].dy, self.cube[x2][y2].dz, ]
#         except:
#             pause = True
#
#         try:
#             cube_observation3 = [self.cube[x3][y3].x, self.cube[x3][y3].y, self.cube[x3][y3].dx,
#                                  self.cube[x3][y3].dy, self.cube[x3][y3].dz, ]
#         except:
#             pause = True
#
#         if x_blank != None:
#             if x_blank == "01":
#                 cube_observation0 = [0, 0, 0, 0, 0]
#                 cube_observation1 = [0, 0, 0, 0, 0]
#             elif x_blank == "23":
#                 cube_observation2 = [0, 0, 0, 0, 0]
#                 cube_observation3 = [0, 0, 0, 0, 0]
#             else:
#                 raise ValueError("horizontal_blank error")
#         if y_blank != None:
#             if y_blank == "02":
#                 cube_observation0 = [0, 0, 0, 0, 0]
#                 cube_observation2 = [0, 0, 0, 0, 0]
#             elif y_blank == "13":
#                 cube_observation1 = [0, 0, 0, 0, 0]
#                 cube_observation3 = [0, 0, 0, 0, 0]
#             else:
#                 raise ValueError("vertical_blank error")
#
#         four_cube_observation = np.array(
#             cube_observation0 + cube_observation1 + cube_observation2 + cube_observation3)
#         return four_cube_observation
#
#     def get_obs_of_aircraft(self, x, y):
#         # 特殊点： 判断是不是边界点，边界点就找不到四个cube了
#         if math.floor(x) == 0 or math.floor(y) == 0 or math.ceil(x) == self.grid_x or math.ceil(
#                 y) == self.grid_y:  # danger
#
#             if x % 1 <= 0.5 and y % 1 <= 0.5:
#                 # 左上
#                 if math.floor(x) == 0 and math.floor(y) == 0:
#                     return self.get_obs_util(x, y, flag=0, x_blank="01", y_blank="02")
#                 elif math.floor(x) == 0:
#                     return self.get_obs_util(x, y, flag=0, y_blank="02")
#                 elif math.floor(y) == 0:
#                     return self.get_obs_util(x, y, flag=0, x_blank="01")
#                 else:
#                     # 左上
#                     four_cube_observation = self.get_obs_util(x, y, flag=0)
#                     return four_cube_observation
#
#             elif x % 1 <= 0.5 and y % 1 >= 0.5:
#                 # 左下
#                 if math.floor(x) == 0 and math.ceil(y) == self.grid_y:
#                     return self.get_obs_util(x, y, flag=1, x_blank="23", y_blank="02")
#                 elif math.floor(x) == 0:
#                     return self.get_obs_util(x, y, flag=1, y_blank="02")
#                 elif math.ceil(y) == self.grid_y:
#                     return self.get_obs_util(x, y, flag=1, x_blank="23")
#                 else:
#                     # 左下
#                     four_cube_observation = self.get_obs_util(x, y, flag=1)
#                     return four_cube_observation
#
#             elif x % 1 >= 0.5 and y % 1 <= 0.5:
#                 # 右上
#                 if math.ceil(x) == self.grid_x and math.floor(y) == 0:
#                     return self.get_obs_util(x, y, flag=2, x_blank="01", y_blank="13")
#                 elif math.ceil(x) == self.grid_x:
#                     return self.get_obs_util(x, y, flag=2, y_blank="13")
#                 elif math.floor(y) == 0:
#                     return self.get_obs_util(x, y, flag=2, x_blank="01")
#                 else:
#                     # 右上
#                     four_cube_observation = self.get_obs_util(x, y, flag=2)
#                     return four_cube_observation
#
#             elif x % 1 >= 0.5 and y % 1 >= 0.5:
#                 # 右下
#                 if math.ceil(x) == self.grid_x and math.ceil(y) == self.grid_y:
#                     return self.get_obs_util(x, y, flag=3, x_blank="23", y_blank="13")
#                 elif math.ceil(x) == self.grid_x:
#                     return self.get_obs_util(x, y, flag=3, y_blank="13")
#                 elif math.ceil(y) == self.grid_y:
#                     return self.get_obs_util(x, y, flag=3, x_blank="23")
#                 else:
#                     # 右下
#                     four_cube_observation = self.get_obs_util(x, y, flag=3)
#                     return four_cube_observation
#         else:
#             # 非特殊点
#             # 首先取余数
#             if x % 1 <= 0.5 and y % 1 <= 0.5:
#                 # 左上
#                 four_cube_observation = self.get_obs_util(x, y, flag=0)
#                 return four_cube_observation
#             elif x % 1 <= 0.5 and y % 1 >= 0.5:
#                 # 左下
#                 four_cube_observation = self.get_obs_util(x, y, flag=1)
#                 return four_cube_observation
#             elif x % 1 >= 0.5 and y % 1 <= 0.5:
#                 # 右上
#                 four_cube_observation = self.get_obs_util(x, y, flag=2)
#                 return four_cube_observation
#             elif x % 1 >= 0.5 and y % 1 >= 0.5:
#                 # 右下
#                 four_cube_observation = self.get_obs_util(x, y, flag=3)
#                 return four_cube_observation
#
#         raise ValueError("get_obs_of_aircraft error")
#         return self.cube
#
#     def check_collision(self, x, y, z):  # 判断是否在cube里面
#         xx, yy = math.floor(x), math.floor(y)
#         if abs(self.cube[xx][yy].x - x) < 0.5 * self.cube[xx][yy].dx and abs(self.cube[xx][yy].y - y) < 0.5 * \
#                 self.cube[xx][yy].dy and z < self.cube[xx][yy].dz:
#             # print("collision")
#             return True
#         else:
#             # x_distance_margin = abs(self.cube[xx][yy].x - x) - 0.5 * self.cube[xx][yy].dx
#             # y_distance_margin = abs(self.cube[xx][yy].y - y) - 0.5 * self.cube[xx][yy].dy
#             # z_distance_margin = z - self.cube[xx][yy].dz
#             return False


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


def plot_linear_cube(ax, x, y, dx, dy, dz, color='red'):
    # 输入xy为模型的中心值
    x = x - 0.5 * dx
    y = y - 0.5 * dy
    z = 0

    xx = [x, x, x + dx, x + dx, x]
    yy = [y, y + dy, y + dy, y, y]
    ax.plot3D(xx, yy, [z] * 5, color=color)
    ax.plot3D(xx, yy, [z + dz] * 5, color=color)
    ax.plot3D([x, x], [y, y], [z, z + dz], color=color)
    ax.plot3D([x, x], [y + dy, y + dy], [z, z + dz], color=color)
    ax.plot3D([x + dx, x + dx], [y + dy, y + dy], [z, z + dz], color=color)
    ax.plot3D([x + dx, x + dx], [y, y], [z, z + dz], color=color)


# # 设定numpy随机数
# np.random.seed(1)  # 这样以来下面的地图就是固定的了吧
# grid_x, grid_y, grid_z = 3, 3, 1  # 设置建筑物的数量
# cube_generator = Cube_generator(grid_x=grid_x, grid_y=grid_y)
# for y in range(grid_y):
#     for x in range(grid_x):
#         cube = cube_generator.cube[x][y]
#         plot_linear_cube(ax, cube.x, cube.y, cube.dx, cube.dy, cube.dz, color='red')
# plt.title('Cube')
# plt.show()


# import time
# dt_obj = datetime.fromtimestamp(time.time())
# formatted_time = dt_obj.strftime('%m-%d %H:%M')


run_time = "{}".format(datetime.today()).replace(":", "-").replace(".", "-")  # '2022-04-24_18-41-59-436547'

grid_x, grid_y, grid_z = 3, 3, 1  # 设置建筑物的数量
seed = 2
number = 1000
hahas = []
with open(f"{grid_x}*{grid_y}_{number}.pkl", "rb") as fo:
    for i in range(number):
        haha = pickle.load(fo)
        hahas.append(haha)


class Env_aricraft(gym.Env):
    def __init__(self, writer, algorithm_name,seed,current_ratio, max_step=1000, static_obs=3):
        # self.state_space = Box(low=-49.0, high=49.0, shape=(6,))
        # self.action_space = Discrete(6)
        self.algorithm_name = algorithm_name
        self.fig_dir = f"{algorithm_name}_fig_ratio_{current_ratio}"
        os.makedirs(f"./{self.fig_dir}/{run_time}")
        self.writer = writer
        self.max_step = max_step
        # self.state_space = Box(low=-49.0, high=49.0, shape=(26+2,),seed=seed)
        self.observation_space = Box(low=-50.0, high=50.0, shape=(26+2,),seed=seed)
        self.action_space = Box(np.array([0.0, 0.0]), np.array([np.pi, 2*np.pi]),seed=seed)
        self.GOAL_POINT = GOAL_POINT
        # self.GOAL_POINT[1],self.GOAL_POINT[2]= 1+48*np.random.rand() , 1+48*np.random.rand()
        # 设定numpy随机数
        # np.random.seed(seed)  # 这样以来下面的地图就是固定的了吧
        # torch.manual_seed(seed)
        cube_generator = Cube_generator(grid_x=grid_x, grid_y=grid_y)
        cube_generator.cube = copy.deepcopy(hahas[0])
        for y in range(grid_y):
            for x in range(grid_x):
                cube = cube_generator.cube[x][y]
                plot_linear_cube(ax, cube.x, cube.y, cube.dx, cube.dy, cube.dz, color='red')

        self.cube_generator = copy.deepcopy(cube_generator)

        self.current_point = copy.deepcopy(START_POINT)
        self.last_point = copy.deepcopy(START_POINT)
        # self.goal = copy.deepcopy(GOAL_POINT)
        # self.start = copy.deepcopy(START_POINT)
        self.load_current()  # todo
        self.current_fre = 10  # todo
        self.current_path = current_path
        self.current_ratio = current_ratio # todo

        self.last_distance = np.sqrt(np.sum((self.current_point - self.GOAL_POINT) ** 2))
        ######OBS########
        # self.obs_num = config.obs_num  # todo
        self.static_obs = static_obs
        # parser.add_argument("--static", type=int, default=0,
        # help="static obs:0,random obs:1,dynamic:2")

        self.obs_n = []
        self.is_collision = False
        self.reach_goal = False
        Obs = [obs1, obs2, obs3, obs4, obs5, obs6, obs7, obs8, obs9, obs10, obs11, obs12, obs13, obs14, obs15, obs16]
        if self.static_obs == 0:
            obs = []
            for i in range(self.obs_num):
                obs_position = np.array(Obs[i])
                self.obs_n.append(Static_obs(obs_position))
            with open(os.path.join(record_dir, 'obs_position.csv'), 'a') as f:
                writer = csv.writer(f)
                writer.writerow(obs)
        if self.static_obs == 2:
            for i in range(self.obs_num):
                obs_position = np.array([0, 0, 0])
                for i in range(3):
                    obs_position[i] = np.random.randint(50)
                self.obs_n.append(Dynnmic_obs(obs_position))
        self.fig_num = 0
        self.episode_step = 0
        self.total_step = 0
        self.episode_reward_0_s = []
        self.episode_rewards_1_s = []
        self.episode_reward_0 = []
        self.episode_reward_1 = []
        self.collision_time = 0
        self.episode = 0
        self.episode_trajuctory_length = 0

    def reset(self, seed=None, options=None):

        # if self.static_obs == 0:
        #     obs = []
        #     for i in range(self.obs_num):
        #         obs_position = np.array([0, 0, 0])
        #         for i in range(3):
        #             obs_position[i] = np.random.randint(50)
        #         self.obs_n.append(Obs(obs_position))
        #     with open(os.path.join(record_dir, 'obs_position.csv'), 'a') as f:
        #         writer = csv.writer(f)
        #         writer.writerow(obs)
        # self.GOAL_POINT[1],self.GOAL_POINT[2]= 1+48*np.random.rand() , 1+48*np.random.rand()
        self.writer.add_scalar("info/self.GOAL_POINT[1]", self.GOAL_POINT[1], self.episode)
        self.writer.add_scalar("info/self.GOAL_POINT[2]", self.GOAL_POINT[2], self.episode)

        Obs = [obs1, obs2, obs3, obs4, obs5, obs6, obs7, obs8, obs9, obs10, obs11, obs12, obs13, obs14, obs15, obs16]
        if self.static_obs == 0:
            obs = []
            for i in range(self.obs_num):
                obs_position = np.array(Obs[i])
                self.obs_n.append(Static_obs(obs_position))
            with open(os.path.join(record_dir, 'obs_position.csv'), 'a') as f:
                writer = csv.writer(f)
                writer.writerow(obs)
        if self.static_obs == 1:
            self.obs_n = []
            obs = []
            for i in range(self.obs_num):
                obs_position = np.array([0, 0, 0])
                for i in range(3):
                    obs_position[i] = np.random.randint(50)
                    obs.append(obs_position[i])
                self.obs_n.append(Obs(obs_position))
            with open(os.path.join(record_dir, 'obs_position.csv'), 'a') as f:
                writer = csv.writer(f)
                writer.writerow(obs)
        if self.static_obs == 2:
            self.obs_n = []
            for i in range(self.obs_num):
                obs_position = np.array([0, 0, 0])
                for i in range(3):
                    obs_position[i] = np.random.randint(50)
                self.obs_n.append(Dynnmic_obs(obs_position))
        self.step_counter = 0
        self.current_point = copy.deepcopy(START_POINT)

        mask = self.caculate_mask()
        # state = np.concatenate((self.goal - self.current_point, mask), axis=0)
        x = self.current_point[0] / 50 * grid_x
        y = self.current_point[1] / 50 * grid_y

        four_cube = self.cube_generator.get_obs_of_aircraft(x=x, y=y)
        four_cube = four_cube / grid_x * 50
        state = np.concatenate((self.current_point, mask, four_cube,self.GOAL_POINT[1:3]), axis=0)
        self.history_x = []
        self.history_y = []
        self.history_z = []
        self.history_length = 0
        self.last_distance = np.sqrt(np.sum((self.current_point - self.GOAL_POINT) ** 2))
        # self.last_current_reward = 0
        self.history_current = 0
        self.is_collision = False
        self.reach_goal = False

        self.episodic_return = 0
        return state
        # return state, {}

    def step(self, action):
        if self.static_obs == 2:
            for i in range(self.obs_num):
                self.obs_n[i].step_move()
        velocity = 1
        self.writer.add_scalar('maybe_ploted/velocity', velocity, self.total_step)
        # action = self.is_action_valid(action)
        position, current_ratio = self.caculate_position(action, velocity)
        # if position[0] > 49 or position[1] > 49 or position[2] > 49:
        #     pause = 1
        #     position, current_ratio = self.caculate_position(action)
        self.last_point = self.current_point
        self.current_point = position
        self.history_x.append(self.current_point[0])
        self.history_y.append(self.current_point[1])
        self.history_z.append(self.current_point[2])
        current = self.caculate_mask()
        # state = np.concatenate((self.goal - self.current_point, current), axis=0)
        x = self.current_point[0] / 50 * grid_x
        y = self.current_point[1] / 50 * grid_y
        four_cube = self.cube_generator.get_obs_of_aircraft(x=x, y=y)
        four_cube = four_cube / grid_x * 50
        state = np.concatenate((self.current_point, current, four_cube,self.GOAL_POINT[1:3]), axis=0)  # todo right？

        # assert self.state_space.contains(state), "%r (%s) invalid" % (state, type(state))

        done = self.is_done(position)
        reward = self.caculate_reward(action, current_ratio, done)

        with open(os.path.join(record_dir, 'train_history_path.csv'), 'a') as f:
            writer = csv.writer(f)
            writer.writerow((self.step_counter, self.current_point[0], self.current_point[1], self.current_point[2]))
        # print(self.current_point)
        self.step_counter = self.step_counter + 1

        # return state, reward, done, self.is_collision, self.reach_goal
        self.episode_step += 1
        self.total_step += 1
        # return state, reward, done, False, {}
        return state, reward, done, {}

    def is_action_valid(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        return action

    def caculate_position(self, action,velocity):
        # current = self.caculate_current()
        # current_ratio = current / self.max_current * self.current_ratio
        current = [1/np.sqrt(2),-1.0/np.sqrt(2),0.0]
        current_random = np.random.normal(size=[3])*0.1
        current_ratio = (current+current_random) * self.current_ratio

        position = np.zeros(3, )
        # for i in range(3):
        #     position[i] = self.current_point[i]
        position = copy.deepcopy(self.current_point)

        # #########lon############
        # if action == 0:
        #     position[0] = position[0] + 1
        # elif action == 1:
        #     position[0] = position[0] - 1
        # #########lat############
        # elif action == 2:
        #     position[1] = position[1] + 1
        # elif action == 3:
        #     position[1] = position[1] - 1
        # #########deep############
        # elif action == 4:
        #     position[2] = position[2] + 1
        # else:
        #     position[2] = position[2] - 1
        theta = action[0]
        phi = action[1]

        delta_x = velocity * np.sin(theta) * np.cos(phi)
        delta_y = velocity * np.sin(theta) * np.sin(phi)
        delta_z = velocity * np.cos(theta)

        position[0] = position[0] + delta_x
        position[1] = position[1] + delta_y
        position[2] = position[2] + delta_z

        position = position + current_ratio
        position = self.is_position_valid(position)
        return position, current_ratio

    def is_position_valid(self, position):
        for i in range(3):
            if position[i] > MAX:
                position[i] = MAX
            if position[i] < MIN:
                position[i] = MIN

        return position

    def caculate_reward(self, action, current_ratio, done):
        if self.reach_goal:
            reward = 100
        elif self.is_collision:
            reward = -100
        else:
            # collision_num = 0
            # for i in range(len(self.obs_n)):
            #     is_collision = self.obs_n[i].is_collision(self.current_point)
            #     if is_collision:
            #         collision_num += 1
            ################distance####################
            distance_forward = np.sqrt(np.sum((self.current_point - self.last_point) ** 2))
            self.episode_trajuctory_length += distance_forward

            distance = np.sqrt(np.sum((self.current_point - self.GOAL_POINT) ** 2))
            distance_reward = distance - self.last_distance
            if distance < self.last_distance:
                # distance_reward = 0.5
                distance_reward = self.last_distance - distance
            else:
                distance_reward = self.last_distance - distance
                # distance_reward = -0.5
            self.last_distance = distance
            # current_reward = self.caculate_current_reward(action, current_ratio)
            # print(self.step_counter, distance_reward, current_reward)
            x = self.current_point[0] / 50 * grid_x
            y = self.current_point[1] / 50 * grid_y
            z = self.current_point[2] / 50 * grid_z
            if self.cube_generator.check_collision(x, y, z):
                self.collision_time += 1
                cube_reward = -5
            else:
                cube_reward = 0.5
            # cube_reward=0
            reward = distance_reward + cube_reward - 1
            self.episodic_return += reward
            self.writer.add_scalar('maybe_ploted/distance_reward', distance_reward, self.total_step)
            self.writer.add_scalar('maybe_ploted/cube_reward', cube_reward, self.total_step)
            self.writer.add_scalar('maybe_ploted/reward', reward, self.total_step)
            self.episode_reward_0.append(distance_reward)
            self.episode_reward_1.append(cube_reward)

        return reward

    def is_done(self, position):
        # x = 4.19 * position[0]
        # y = 4.49 * position[1]
        # z = position[2]
        # xx = np.floor(x)
        # yy = np.floor(y)
        # zz = 70.9663 * z - 3477.3549
        # a = self.h[int(xx)][int(yy)]

        # if self.h[int(xx)][int(yy)] > zz:#todo ！！！
        #     self.is_collision = True

        # if -1163.2*x+1211.6*y-1679.1*z+19916<0:
        #     self.is_collision = True
        # if 98.9068*x+292.2954*y-228.9693*z-10775<0:
        #     self.is_collision = True
        # if z<0.9909:
        #     self.is_collision = True
        # for i in range(3):
        #     if position[i] > MAX:
        #         self.is_collision = True
        #     if position[i] < MIN:
        #         self.is_collision = True
        for i in range(len(self.obs_n)):
            is_collision = self.obs_n[i].is_collision(self.current_point)
            if is_collision:
                self.is_collision = True
        if np.sqrt(np.sum((self.current_point - self.GOAL_POINT) ** 2)) < 3:
            self.reach_goal = True
            self.success = 1
        else:
            self.success = 0
        self.step_flag = False
        if self.episode_step > self.max_step:
            self.step_flag = True
        if self.reach_goal or self.is_collision or self.step_flag:
            self.episode_reward_0_s.append(self.episode_reward_0)
            self.episode_rewards_1_s.append(self.episode_reward_1)
            self.visualization()


            print("self.collision_time:", self.collision_time)
            self.writer.add_scalar("to_be_ploted/self.episode", self.episode, self.episode)
            self.writer.add_scalar("to_be_ploted/collision_time", self.collision_time, self.episode)
            self.writer.add_scalar("to_be_ploted/steps", self.episode_step, self.episode)
            self.writer.add_scalar("to_be_ploted/success", self.success, self.episode)
            print("to_be_ploted/episodic_return_true:", self.episodic_return)
            self.writer.add_scalar("to_be_ploted/episodic_return_true", self.episodic_return, self.episode)
            self.writer.add_scalar("to_be_ploted/episode_trajuctory_length", self.episode_trajuctory_length,
                                   self.episode)
            self.episode_reward_0 = []
            self.episode_reward_1 = []
            # wandb.log({"done/episode": self.episode_step})
            self.episode_step = 0
            self.episode += 1
            self.episodic_return = 0

            self.episode_trajuctory_length = 0
            self.collision_time = 0

            with open(f"trajectory_{self.algorithm_name}.pkl","a+b") as fo:
                pickle.dump([self.history_x,self.history_y,self.history_z],fo)

            return True
        else:
            return False

    # plt.cla()
    # fig = plt.figure(figsize=(8, 6))
    # ax = plt.subplot(111)
    # plt.plot([i for i in range(len(self.episode_reward_0))], self.episode_reward_0)
    # plt.show()
    def visualization(self):
        plt.cla()
        ax = plt.subplot(111, projection='3d')
        elev = -160
        azim = 230 - 90
        ax.view_init(elev, azim)
        ax.invert_zaxis()
        plt.rcParams['figure.figsize'] = (12.0, 12.0)

        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # cube_generator = self.cube_generator
        for y in range(grid_y):
            for x in range(grid_x):
                cube = self.cube_generator.cube[x][y]
                plot_linear_cube(ax, cube.x / grid_x * 50, cube.y / grid_y * 50, cube.dx / grid_x * 50,
                                 cube.dy / grid_y * 50,
                                 cube.dz * 50, color='red')
        # plt.show()
        # x, y, z = np.meshgrid(np.arange(0, 50, 1),
        #                       np.arange(0, 50, 1),
        #                       np.arange(0, 50, 1))
        # ax.quiver(x, y, z, self.u, self.v, self.w, length=1)

        ax.plot(START_POINT[0], START_POINT[1], START_POINT[2], 'go', markersize=7, markeredgecolor='k')
        ax.plot(self.history_x, self.history_y, self.history_z, '-b', markersize=5, markeredgecolor='k')
        ax.plot(GOAL_POINT[0], GOAL_POINT[1], GOAL_POINT[2], 'ro', markersize=7, markeredgecolor='k')
        plt.savefig(f"./{self.fig_dir}/{run_time}/{self.fig_num}.png")
        self.fig_num += 1
        plt.title(f"v1_{self.fig_dir}")
        # plt.show()
        # plt.pause(0.001)
        plt.cla()

    def load_current(self):
        self.u = np.zeros((25, 50, 50, 50))
        self.v = np.zeros((25, 50, 50, 50))
        self.w = np.zeros((25, 50, 50, 50))
        self.k = np.zeros((25, 50, 50, 50))
        self.h = np.zeros((207, 222))  # todo dai 这个是干啥的亚
        hfile = current_path + hcurrent
        hdic = sc.loadmat(hfile)
        hdata = hdic['h']
        self.h = hdata
        for i in range(25):
            ufile = current_path + str(i + 1) + ucurrent
            udic = sc.loadmat(ufile)
            udata = udic['Current_u']
            vfile = current_path + str(i + 1) + vcurrent
            vdic = sc.loadmat(vfile)
            vdata = vdic['Current_v']
            wfile = current_path + str(i + 1) + wcurrent
            wdic = sc.loadmat(wfile)
            wdata = wdic['Current_w']
            kfile = current_path + str(i + 1) + kcurrent
            kdic = sc.loadmat(kfile)
            kdata = kdic['Current_k']
            self.u[i, :, :, :] = udata
            self.v[i, :, :, :] = vdata
            self.w[i, :, :, :] = wdata
            self.k[i, :, :, :] = kdata
        self.max_current = np.max(self.k)

    def caculate_current(self):
        current_frame = self.step_counter // self.current_fre + 1
        # rint('current_frame:', current_frame )
        if current_frame > 25:
            current_frame = current_frame % 25
            if current_frame == 0:
                current_frame = 25
        current_frame = current_frame - 1
        # print('current data:', current_frame)
        position = self.current_point
        up = []
        down = []
        for i in range(3):
            upup = np.ceil(position[i])
            up.append(upup)
            downdown = np.floor(position[i])
            down.append(downdown)
        point1 = [up[0], down[1], down[2]]
        point2 = [up[0], up[1], down[2]]
        point3 = [down[0], up[1], down[2]]
        point4 = [down[0], down[1], down[2]]
        point5 = [up[0], up[1], up[2]]
        point6 = [up[0], down[1], up[2]]
        point7 = [down[0], down[1], up[2]]
        point8 = [down[0], up[1], up[2]]

        dis1 = np.sqrt(np.sum((position - np.array(point1)) ** 2))
        dis2 = np.sqrt(np.sum((position - np.array(point2)) ** 2))
        dis3 = np.sqrt(np.sum((position - np.array(point3)) ** 2))
        dis4 = np.sqrt(np.sum((position - np.array(point4)) ** 2))
        dis5 = np.sqrt(np.sum((position - np.array(point5)) ** 2))
        dis6 = np.sqrt(np.sum((position - np.array(point6)) ** 2))
        dis7 = np.sqrt(np.sum((position - np.array(point7)) ** 2))
        dis8 = np.sqrt(np.sum((position - np.array(point8)) ** 2))
        sum_dis = dis1 + dis2 + dis3 + dis4 + dis5 + dis6 + dis7 + dis8

        if sum_dis == 0:
            current_u = self.u[current_frame, int(point1[1]), int(point1[0]), int(point1[2])]
            current_v = self.v[current_frame, int(point1[1]), int(point1[0]), int(point1[2])]
            current_w = self.w[current_frame, int(point1[1]), int(point1[0]), int(point1[2])]
        else:
            ##########u#####################
            current_u1 = self.u[current_frame, int(point1[1]), int(point1[0]), int(point1[2])]
            current_u2 = self.u[current_frame, int(point2[1]), int(point2[0]), int(point2[2])]
            current_u3 = self.u[current_frame, int(point3[1]), int(point3[0]), int(point3[2])]
            current_u4 = self.u[current_frame, int(point4[1]), int(point4[0]), int(point4[2])]
            current_u5 = self.u[current_frame, int(point5[1]), int(point5[0]), int(point5[2])]
            current_u6 = self.u[current_frame, int(point6[1]), int(point6[0]), int(point6[2])]
            current_u7 = self.u[current_frame, int(point7[1]), int(point7[0]), int(point7[2])]
            current_u8 = self.u[current_frame, int(point8[1]), int(point8[0]), int(point8[2])]
            current_u = (current_u1 * dis1 + current_u2 * dis2 + current_u3 * dis3 + current_u4 * dis4 \
                         + current_u5 * dis5 + current_u6 * dis6 + current_u7 * dis7 + current_u8 * dis8) / sum_dis

            ##########v#####################
            current_v1 = self.v[current_frame, int(point1[1]), int(point1[0]), int(point1[2])]
            current_v2 = self.v[current_frame, int(point2[1]), int(point2[0]), int(point2[2])]
            current_v3 = self.v[current_frame, int(point3[1]), int(point3[0]), int(point3[2])]
            current_v4 = self.v[current_frame, int(point4[1]), int(point4[0]), int(point4[2])]
            current_v5 = self.v[current_frame, int(point5[1]), int(point5[0]), int(point5[2])]
            current_v6 = self.v[current_frame, int(point6[1]), int(point6[0]), int(point6[2])]
            current_v7 = self.v[current_frame, int(point7[1]), int(point7[0]), int(point7[2])]
            current_v8 = self.v[current_frame, int(point8[1]), int(point8[0]), int(point8[2])]
            current_v = (current_v1 * dis1 + current_v2 * dis2 + current_v3 * dis3 + current_v4 * dis4 \
                         + current_v5 * dis5 + current_v6 * dis6 + current_v7 * dis7 + current_v8 * dis8) / sum_dis
            ##########w#####################
            current_w1 = self.w[current_frame, int(point1[1]), int(point1[0]), int(point1[2])]
            current_w2 = self.w[current_frame, int(point2[1]), int(point2[0]), int(point2[2])]
            current_w3 = self.w[current_frame, int(point3[1]), int(point3[0]), int(point3[2])]
            current_w4 = self.w[current_frame, int(point4[1]), int(point4[0]), int(point4[2])]
            current_w5 = self.w[current_frame, int(point5[1]), int(point5[0]), int(point5[2])]
            current_w6 = self.w[current_frame, int(point6[1]), int(point6[0]), int(point6[2])]
            current_w7 = self.w[current_frame, int(point7[1]), int(point7[0]), int(point7[2])]
            current_w8 = self.w[current_frame, int(point8[1]), int(point8[0]), int(point8[2])]
            current_w = (current_w1 * dis1 + current_w2 * dis2 + current_w3 * dis3 + current_w4 * dis4 \
                         + current_w5 * dis5 + current_w6 * dis6 + current_w7 * dis7 + current_w8 * dis8) / sum_dis
        current = [current_u, current_v, current_w]
        return current

    def caculate_mask(self):
        current = self.caculate_current()
        current_ratio = current / self.max_current * self.current_ratio
        mask = current_ratio
        # r = np.sqrt(np.sum(np.power(current_ratio,2)))
        # if r==0:
        #     mask = np.array([0.0,0.0,0.0])
        # else:
        #     theta = np.arccos(current_ratio[2]/(r))
        #     phi = np.arctan2(current_ratio[1],current_ratio[0])
        #     mask = np.array([r,theta,phi])
        return mask

    def caculate_current_reward(self, action, current_ratio):
        if action == 0:
            alpha = np.pi / 2 - np.arctan2(current_ratio[0], np.sqrt(current_ratio[1] ** 2 + current_ratio[2] ** 2))
            angle = 0
            current_reward = np.cos(alpha - angle)
        elif action == 1:
            alpha = np.pi / 2 - np.arctan2(current_ratio[0], np.sqrt(current_ratio[1] ** 2 + current_ratio[2] ** 2))
            angle = np.pi
            current_reward = np.cos(alpha - angle)
        elif action == 2:
            beta = np.pi / 2 - np.arctan2(current_ratio[1], np.sqrt(current_ratio[0] ** 2 + current_ratio[2] ** 2))
            angle = 0
            current_reward = np.cos(beta - angle)
        elif action == 3:
            beta = np.pi / 2 - np.arctan2(current_ratio[1], np.sqrt(current_ratio[0] ** 2 + current_ratio[2] ** 2))
            angle = np.pi
            current_reward = np.cos(beta - angle)
        elif action == 4:
            gamma = np.pi / 2 - np.arctan2(current_ratio[2], np.sqrt(current_ratio[0] ** 2 + current_ratio[1] ** 2))
            angle = 0
            current_reward = np.cos(gamma - angle)
        else:
            gamma = np.pi / 2 - np.arctan2(current_ratio[2], np.sqrt(current_ratio[0] ** 2 + current_ratio[1] ** 2))
            angle = np.pi
            current_reward = np.cos(gamma - angle)
        self.history_current = self.history_current + current_reward
        return current_reward

    def if_done_history_length(self):
        length = 0
        for i in range(len(self.history_x) - 1):
            dis = np.sqrt(
                np.power((self.history_x[i] - self.history_x[i + 1]), 2) + np.power(
                    (self.history_y[i] - self.history_y[i + 1]), 2) +
                np.power((self.history_z[i] - self.history_z[i + 1]), 2))
            length = length + dis
        return length

    def if_done_history_current(self):
        return self.history_current

    def save_path_and_figure(self, episode):
        with open(os.path.join(record_dir, 'train_path.csv'), 'a') as f:
            writer = csv.writer(f)
            for i in range(len(self.history_x)):
                writer.writerow((episode, i, self.history_x[i], self.history_y[i], self.history_z[i]))
        elev = -160
        azim = 230
        plt.rcParams['figure.figsize'] = (12.0, 12.0)
        ax = plt.subplot(111, projection='3d')
        # x, y, z = np.meshgrid(np.arange(0, 50, 1),
        #                       np.arange(0, 50, 1),
        #                       np.arange(0, 50, 1))
        # ax.quiver(x, y, z, self.u, self.v, self.w, length=1)
        ax.view_init(elev, azim)
        ax.invert_zaxis()
        for i in range(len(self.obs_n)):
            ax.plot(self.obs_n[i].position[0], self.obs_n[i].position[1], self.obs_n[i].position[2], 'yo', markersize=7,
                    markeredgecolor='k')
        ax.plot(START_POINT[0], START_POINT[1], START_POINT[2], 'go', markersize=7, markeredgecolor='k')
        ax.plot(self.history_x, self.history_y, self.history_z, 'bo', markersize=5, markeredgecolor='k')
        ax.plot(GOAL_POINT[0], GOAL_POINT[1], GOAL_POINT[2], 'ro', markersize=7, markeredgecolor='k')

        plt.savefig(figure_dir + '/' + str(episode) + 'history.png')
        plt.cla()

    def test_path_and_figure(self, episode):
        with open(os.path.join(test_dir, 'test_path.csv'), 'a') as f:
            writer = csv.writer(f)
            for i in range(len(self.history_x)):
                writer.writerow((episode, i, self.history_x[i], self.history_y[i], self.history_z[i]))
        elev = -160
        azim = 230
        plt.rcParams['figure.figsize'] = (12.0, 12.0)
        ax = plt.subplot(111, projection='3d')
        # x, y, z = np.meshgrid(np.arange(0, 50, 1),
        #                       np.arange(0, 50, 1),
        #                       np.arange(0, 50, 1))
        # ax.quiver(x, y, z, self.u, self.v, self.w, length=1)
        ax.view_init(elev, azim)
        ax.invert_zaxis()
        for i in range(len(self.obs_n)):
            ax.plot(self.obs_n[i].position[0], self.obs_n[i].position[1], self.obs_n[i].position[2], 'yo', markersize=7,
                    markeredgecolor='k')
        ax.plot(START_POINT[0], START_POINT[1], START_POINT[2], 'go', markersize=7, markeredgecolor='k')
        ax.plot(self.history_x, self.history_y, self.history_z, 'bo', markersize=5, markeredgecolor='k')
        ax.plot(GOAL_POINT[0], GOAL_POINT[1], GOAL_POINT[2], 'ro', markersize=7, markeredgecolor='k')

        plt.savefig(test_figure + '/' + str(episode) + 'history.png')
        plt.cla()
