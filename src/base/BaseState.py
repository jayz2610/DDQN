from src.Map.Map import Map
import numpy as np


class BaseState:
    def __init__(self, map_init: Map):
        self.no_fly_zone = map_init.nfz
        self.obstacles = map_init.obstacles
        self.landing_zone = map_init.start_land_zone

    @property
    def shape(self):
        return self.landing_zone.shape[:2]  # 返回前两个维度，表示地图的行数和列数。

    def get_scalars(self, give_position=False):
        """
        Base implementation of get_scalars.
        In a more complex implementation, you could add the current state properties.
        """
        scalars = [np.sum(self.no_fly_zone), np.sum(self.obstacles)]  # Example scalars

        if give_position:
            # Optionally return position or related data if requested
            scalars.extend([0, 0])  # Placeholder for position data (you may need to adjust based on your state)

        return scalars
