import copy
import math
import pickle

import numpy as np
import torch


class One_cube():
    def __init__(self, x, y, dx=None, dy=None, dz=None):  # 这里的x,y是中心点的坐标,并且中心都是1.5 3.5 等
        self.x = x
        self.y = y
        if dx is None and dy is None and dz is None:
            self.dx = (np.random.rand() + 0.2) / 1.2
            self.dy = (np.random.rand() + 0.2) / 1.2
            # self.dz = 2 * (np.random.rand() + 0.2) / 1.2 # 这里设置为2 让他更像一栋楼，但是
            self.dz = (np.random.rand() + 0.2) / 1.2
        else:
            self.dx = dx
            self.dy = dy
            self.dz = dz


class Cube_generator():
    def __init__(self, grid_x=10, grid_y=10):
        self.grid_x = grid_x
        self.grid_y = grid_y

        matrix = []
        for i in range(grid_x):
            row = []
            for j in range(grid_y):
                row.append(0)  # 添加初始值，这里使用0
            matrix.append(row)
        self.cube = matrix

        for y in range(self.grid_y):
            for x in range(self.grid_x):
                one_cube = One_cube(x + 0.5, y + 0.5)
                self.check_valid(one_cube, x, y)
                self.cube[x][y] = one_cube
        self.cube_num = 0

    def check_valid(self, onecube, x, y):
        if x > 0:
            assert abs(onecube.x - self.cube[x - 1][y].x) > 0.5 * abs(onecube.dx + self.cube[x - 1][y].dx)
        if y > 0:
            assert abs(onecube.y - self.cube[x][y - 1].y) > 0.5 * abs(onecube.dy + self.cube[x][y - 1].dy)

    def get_obs_util(self, x, y, flag, x_blank=None, y_blank=None):
        x, y = math.floor(x), math.floor(y)
        if flag == 0:
            # 左上
            x0, y0 = x - 1, y - 1
            x1, y1 = x, y - 1
            x2, y2 = x - 1, y
            x3, y3 = x, y
        elif flag == 1:
            # 左下
            x0, y0 = x - 1, y
            x1, y1 = x, y
            x2, y2 = x - 1, y + 1
            x3, y3 = x, y + 1
        elif flag == 2:
            # 右上
            x0, y0 = x, y - 1
            x1, y1 = x + 1, y - 1
            x2, y2 = x, y
            x3, y3 = x + 1, y
        elif flag == 3:
            # 右下
            x0, y0 = x, y
            x1, y1 = x + 1, y
            x2, y2 = x, y + 1
            x3, y3 = x + 1, y + 1
        else:
            raise ValueError("flag error")

        try:
            cube_observation0 = [self.cube[x0][y0].x, self.cube[x0][y0].y, self.cube[x0][y0].dx,
                                 self.cube[x0][y0].dy, self.cube[x0][y0].dz, ]
        except:
            pause = True

        try:
            cube_observation1 = [self.cube[x1][y1].x, self.cube[x1][y1].y, self.cube[x1][y1].dx,
                                 self.cube[x1][y1].dy, self.cube[x1][y1].dz, ]
        except:
            pause = True

        try:
            cube_observation2 = [self.cube[x2][y2].x, self.cube[x2][y2].y, self.cube[x2][y2].dx,
                                 self.cube[x2][y2].dy, self.cube[x2][y2].dz, ]
        except:
            pause = True

        try:
            cube_observation3 = [self.cube[x3][y3].x, self.cube[x3][y3].y, self.cube[x3][y3].dx,
                                 self.cube[x3][y3].dy, self.cube[x3][y3].dz, ]
        except:
            pause = True

        if x_blank != None:
            if x_blank == "01":
                cube_observation0 = [0, 0, 0, 0, 0]
                cube_observation1 = [0, 0, 0, 0, 0]
            elif x_blank == "23":
                cube_observation2 = [0, 0, 0, 0, 0]
                cube_observation3 = [0, 0, 0, 0, 0]
            else:
                raise ValueError("horizontal_blank error")
        if y_blank != None:
            if y_blank == "02":
                cube_observation0 = [0, 0, 0, 0, 0]
                cube_observation2 = [0, 0, 0, 0, 0]
            elif y_blank == "13":
                cube_observation1 = [0, 0, 0, 0, 0]
                cube_observation3 = [0, 0, 0, 0, 0]
            else:
                raise ValueError("vertical_blank error")

        four_cube_observation = np.array(
            cube_observation0 + cube_observation1 + cube_observation2 + cube_observation3)
        return four_cube_observation

    def get_obs_of_aircraft(self, x, y):
        # 特殊点： 判断是不是边界点，边界点就找不到四个cube了
        if math.floor(x) == 0 or math.floor(y) == 0 or math.ceil(x) == self.grid_x or math.ceil(
                y) == self.grid_y:  # danger

            if x % 1 <= 0.5 and y % 1 <= 0.5:
                # 左上
                if math.floor(x) == 0 and math.floor(y) == 0:
                    return self.get_obs_util(x, y, flag=0, x_blank="01", y_blank="02")
                elif math.floor(x) == 0:
                    return self.get_obs_util(x, y, flag=0, y_blank="02")
                elif math.floor(y) == 0:
                    return self.get_obs_util(x, y, flag=0, x_blank="01")
                else:
                    # 左上
                    four_cube_observation = self.get_obs_util(x, y, flag=0)
                    return four_cube_observation

            elif x % 1 <= 0.5 and y % 1 >= 0.5:
                # 左下
                if math.floor(x) == 0 and math.ceil(y) == self.grid_y:
                    return self.get_obs_util(x, y, flag=1, x_blank="23", y_blank="02")
                elif math.floor(x) == 0:
                    return self.get_obs_util(x, y, flag=1, y_blank="02")
                elif math.ceil(y) == self.grid_y:
                    return self.get_obs_util(x, y, flag=1, x_blank="23")
                else:
                    # 左下
                    four_cube_observation = self.get_obs_util(x, y, flag=1)
                    return four_cube_observation

            elif x % 1 >= 0.5 and y % 1 <= 0.5:
                # 右上
                if math.ceil(x) == self.grid_x and math.floor(y) == 0:
                    return self.get_obs_util(x, y, flag=2, x_blank="01", y_blank="13")
                elif math.ceil(x) == self.grid_x:
                    return self.get_obs_util(x, y, flag=2, y_blank="13")
                elif math.floor(y) == 0:
                    return self.get_obs_util(x, y, flag=2, x_blank="01")
                else:
                    # 右上
                    four_cube_observation = self.get_obs_util(x, y, flag=2)
                    return four_cube_observation

            elif x % 1 >= 0.5 and y % 1 >= 0.5:
                # 右下
                if math.ceil(x) == self.grid_x and math.ceil(y) == self.grid_y:
                    return self.get_obs_util(x, y, flag=3, x_blank="23", y_blank="13")
                elif math.ceil(x) == self.grid_x:
                    return self.get_obs_util(x, y, flag=3, y_blank="13")
                elif math.ceil(y) == self.grid_y:
                    return self.get_obs_util(x, y, flag=3, x_blank="23")
                else:
                    # 右下
                    four_cube_observation = self.get_obs_util(x, y, flag=3)
                    return four_cube_observation
        else:
            # 非特殊点
            # 首先取余数
            if x % 1 <= 0.5 and y % 1 <= 0.5:
                # 左上
                four_cube_observation = self.get_obs_util(x, y, flag=0)
                return four_cube_observation
            elif x % 1 <= 0.5 and y % 1 >= 0.5:
                # 左下
                four_cube_observation = self.get_obs_util(x, y, flag=1)
                return four_cube_observation
            elif x % 1 >= 0.5 and y % 1 <= 0.5:
                # 右上
                four_cube_observation = self.get_obs_util(x, y, flag=2)
                return four_cube_observation
            elif x % 1 >= 0.5 and y % 1 >= 0.5:
                # 右下
                four_cube_observation = self.get_obs_util(x, y, flag=3)
                return four_cube_observation

        raise ValueError("get_obs_of_aircraft error")
        return self.cube

    def check_collision(self, x, y, z):  # 判断是否在cube里面
        xx, yy = math.floor(x), math.floor(y)
        if abs(self.cube[xx][yy].x - x) < 0.5 * self.cube[xx][yy].dx and abs(self.cube[xx][yy].y - y) < 0.5 * \
                self.cube[xx][yy].dy and z < self.cube[xx][yy].dz:
            # print("collision")
            return True
        else:
            # x_distance_margin = abs(self.cube[xx][yy].x - x) - 0.5 * self.cube[xx][yy].dx
            # y_distance_margin = abs(self.cube[xx][yy].y - y) - 0.5 * self.cube[xx][yy].dy
            # z_distance_margin = z - self.cube[xx][yy].dz
            return False

if __name__ == '__main__':

    grid_x, grid_y, grid_z = 3, 3, 1  # 设置建筑物的数量
    seed = 2
    number = 1000

    np.random.seed(seed)  # 这样以来下面的地图就是固定的了吧
    torch.manual_seed(seed)
    for i in range(number):
        cube_generator = Cube_generator(grid_x=grid_x, grid_y=grid_y)
        with open(f"{grid_x}*{grid_y}_{number}.pkl", "ab") as fo:
            pickle.dump(cube_generator.cube, fo)
    copy.deepcopy(cube_generator)

    hahas = []
    with open(f"{grid_x}*{grid_y}_{number}.pkl", "rb") as fo:
        for i in range(number):
            haha = pickle.load(fo)
            hahas.append(haha)
    haha = True
