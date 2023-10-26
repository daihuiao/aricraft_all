import random

import numpy as np

grid_x = 3
grid_y = 3


class obstacle_for_cube():
    def __init__(self, number_of_obstacle, radius):
        obstacle_list = []
        numbers = list(range(1, (grid_x + 1) * (grid_y + 1) + 1))  # 创建包含1到16之间所有数的列表
        random_numbers = random.sample(numbers, number_of_obstacle)  # 从列表中随机选择两个数
        random_index = 0
        while (len(obstacle_list) < number_of_obstacle):
            x = (random_numbers[random_index] - 1) % 4
            y = (random_numbers[random_index] - 1) // 4
            z = random.random()
            obstacle_list.append([x, y, z])
            random_index += 1
        self.obstacle_list = np.array(obstacle_list)
        self.radius = radius

    def check_collision(self, x, y, z):

        distance = np.linalg.norm(self.obstacle_list - np.array([x, y, z]), axis=1)
        for i in distance:
            if i < self.radius:
                return True

        return False

    def get_obstacle(self,x,y,z):
        distance = np.linalg.norm(self.obstacle_list - np.array([x, y, z]), axis=1)
        nearest_obstacle_index = np.argmin(distance)
        print(nearest_obstacle_index)
        return self.obstacle_list[nearest_obstacle_index]

obstacle_for_cube = obstacle_for_cube(10, 0.5)
obstacle_for_cube.check_collision(0, 0, 0)
obstacle_for_cube.get_obstacle(0,0,0)