import numpy as np
from skimage import io
import heapq
import random


class Map:
    def __init__(self, map_data):
        self.start_land_zone = map_data[:, :, 2].astype(bool)
        self.nfz = map_data[:, :, 0].astype(bool)
        self.obstacles = map_data[:, :, 1].astype(bool)

    def a_star(self, start, goal):
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}

        while open_set:
            current = heapq.heappop(open_set)[1]

            if current == goal:
                return self.reconstruct_path(came_from, current)

            for neighbor in self.get_neighbors(current):
                tentative_g = g_score[current] + 1  # 假设每步成本为1
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        return None  # 无路径

    def get_random_landing_position(self):
        """Return a random position from the landing zone."""
        starting_vector = self.get_starting_vector()  # Get all available landing positions
        if starting_vector:
            return random.choice(starting_vector)  # Randomly select a position
        else:
            raise ValueError("No valid landing positions available.")

    def heuristic(self, a, b):
        """曼哈顿距离作为启发式函数"""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    # Inside Map class
    def get_neighbors(self, pos):
        """获取当前位置的可行邻居（避开障碍物和 NFZ）"""  # Modified docstring
        neighbors = []
        h, w = self.obstacles.shape[:2]  # Get map dimensions
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            x, y = pos[0] + dx, pos[1] + dy
            # Check bounds first
            if 0 <= x < w and 0 <= y < h:
                # Check BOTH obstacles AND nfz
                if not self.obstacles[y, x] and not self.nfz[y, x]:  # ADDED NFZ CHECK
                    neighbors.append((x, y))
        return neighbors

    def reconstruct_path(self, came_from, current):
        """重建路径"""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path[::-1]  # 反转路径

    def get_starting_vector(self):
        similar = np.where(self.start_land_zone)
        return list(zip(similar[1], similar[0]))

    def get_free_space_vector(self):
        free_space = np.logical_not(
            np.logical_or(self.obstacles, self.start_land_zone))
        free_idcs = np.where(free_space)
        return list(zip(free_idcs[1], free_idcs[0]))

    def get_size(self):
        return self.start_land_zone.shape[:2]


def load_image(path):
    if type(path) is not str:
        raise TypeError('path needs to be a string')
    data = io.imread(path, as_gray=True)
    return np.array(data, dtype=bool)


def save_image(path, image):
    if type(path) is not str:
        raise TypeError('path needs to be a string')
    if image.dtype == bool:
        io.imsave(path, image * np.uint8(255))
    else:
        io.imsave(path, image)


def load_map(path):
    if type(path) is not str:
        raise TypeError('path needs to be a string')
    data = io.imread(path, as_gray=False)
    return Map(data)


def load_target(path):
    if type(path) is not str:
        raise TypeError('path needs to be a string')
    data = io.imread(path, as_gray=True)
    return np.array(data, dtype=bool)

