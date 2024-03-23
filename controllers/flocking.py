import numpy as np
from typing import Dict, List, Tuple
import json

###############################################################################
# Controller
###############################################################################

class Controller:
    def __init__(self, config_path):
        
        config = self._load_config_from_json(config_path)
        self.config = config["controller"]

        self.n_agent = config["num_agents"]
        self.alpha = self.config["alpha"]
        self.beta = self.config["beta"]
        self.epsilon = self.config["epsilon"]
        self.sigmas = np.full(self.n_agent, self.config["sigma_const"])
        self.sensing_range = self.config["sensing_range"]
        self.h_alignment = self.config["h_alignment"]
        self.k1 = self.config["k1"]
        self.k2 = self.config["k2"]
        self.umax_const = self.config["umax_const"]
        self.wmax = self.config["wmax"]

    def _load_config_from_json(self, config_path: str) -> Dict:
        """
        Loads the configuration from a JSON file.

        Args:
            config_path (str): Path to the JSON configuration file.

        Returns:
            dict: Configuration parameters.
        """
        with open(config_path, 'r') as file:
            config = json.load(file)
        return config
    
    def calculate_forces(self, pos_xs, pos_ys, pos_hs, boundary=20):
        xx1, xx2 = np.meshgrid(pos_xs, pos_xs)
        yy1, yy2 = np.meshgrid(pos_ys, pos_ys)
        
        d_ij_x = xx1 - xx2
        d_ij_y = yy1 - yy2

        d_ij_x = np.where(d_ij_x > boundary / 2, boundary - d_ij_x, d_ij_x)
        d_ij_x = np.where(d_ij_x <= -boundary / 2, boundary + d_ij_x, d_ij_x)
        
        d_ij_y = np.where(d_ij_y > boundary / 2, boundary - d_ij_y, d_ij_y)
        d_ij_y = np.where(d_ij_y <= -boundary / 2, boundary + d_ij_y, d_ij_y)

        d_ij = np.sqrt(np.multiply(d_ij_x, d_ij_x) + np.multiply(d_ij_y, d_ij_y))
        
        def wrap_angle(angle):
            return np.arctan2(np.sin(angle), np.cos(angle))

        d_ij[d_ij > self.sensing_range] = np.inf
        d_ij[d_ij == 0.0] = np.inf
        # ij_ang = np.arctan2(d_ij_y, d_ij_x) - pos_hs[:, np.newaxis]
        ij_ang = np.arctan2(d_ij_y, d_ij_x) - pos_hs[:, np.newaxis]
        ij_ang = wrap_angle(ij_ang)
        forces = -self.epsilon * (2 * (self.sigmas[:, np.newaxis] ** 4 / d_ij ** 5) - (self.sigmas[:, np.newaxis] ** 2 / d_ij ** 3))
        forces[d_ij == np.inf] = 0.0
        forces_alignment = np.sum(pos_hs) - pos_hs

        return d_ij_x, d_ij_y, d_ij, ij_ang, forces, forces_alignment
    
    def calculate_force_xy(self, forces, ij_ang, forces_alignment):
        f_x = self.alpha * np.sum(np.multiply(forces, np.cos(ij_ang)), axis=1) + \
              int(self.h_alignment) * self.beta * np.cos(forces_alignment)
        f_y = self.alpha * np.sum(np.multiply(forces, np.sin(ij_ang)), axis=1) + \
              int(self.h_alignment) * self.beta * np.sin(forces_alignment)
        return f_x, f_y
    
    def calculate_u_w(self, f_x, f_y):
        f_mag = np.sqrt(np.square(f_x) + np.square(f_y))
        glob_ang = np.arctan2(f_y, f_x)

        u = self.k1 * np.multiply(f_mag, np.cos(glob_ang)) + 0.05
        u[u > self.umax_const] = self.umax_const
        u[u < 0] = 0.0

        w = self.k2 * np.multiply(f_mag, np.sin(glob_ang))
        w[w > self.wmax] = self.wmax
        w[w < -self.wmax] = -self.wmax

        return u, w
    
    def calculate_control(self, observations):
        pos_xs, pos_ys, pos_hs = observations['Positions'].T

        d_ij_x, d_ij_y, d_ij, ij_ang, forces, forces_alignment = self.calculate_forces(pos_xs, pos_ys, pos_hs)
        
        f_x, f_y = self.calculate_force_xy(forces, ij_ang, forces_alignment)
        u, w = self.calculate_u_w(f_x, f_y)
        return np.stack((u, w))