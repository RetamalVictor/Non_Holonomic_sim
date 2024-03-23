from pprint import pprint
import gym, json
import numpy as np
from gym import spaces
import matplotlib.pyplot as plt
from copy import copy
import sys
import os
sys.path.append(os.getcwd())

import json
from typing import Dict, List, Tuple
from rewards.reward_component import RewardInitializer


class ActiveElasticFlockingEnv(gym.Env):
    """
    A reinforcement learning environment for active elastic flocking.

    Attributes:
        config (dict): Configuration parameters for the environment.
        num_agents (int): Number of agents in the environment.
        collision_distance (float): Distance at which agents are considered to have collided.
        sensor_range (float): Sensing range of the agents.
        k (int): Number of nearest neighbors considered for each agent.
        boundary (float): Boundary of the environment.
        dt (float): Time step for the simulation.
        render_bool (bool): Indicates whether to render the environment.
        observation_space (List[spaces.Box]): Observation space of the environment.
        action_space (List[spaces.Box]): Action space of the environment.
        velocities (np.ndarray): Velocities of the agents.
        agents (List[np.ndarray]): Positions and headings of the agents.
        orders (List[float]): Order parameter values over time.
        spacing (float): Spacing between agents in the initial configuration.
        init_x (float): Initial x-coordinate for agent positions.
        init_y (float): Initial y-coordinate for agent positions.
    """
    def __init__(self, config_path: str):
        """
        Initializes the ActiveElasticFlockingEnv environment.

        Args:
            config_path (str): Path to the JSON configuration file.
        """
        super(ActiveElasticFlockingEnv, self).__init__()
        config = self._load_config_from_json(config_path)
        self._extract_params(config)
        self._init_rewards(config["reward_components"], config["reward_params"])
        self.reset()

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

    def _extract_params(self, config: dict):
        """
        Extracts parameters from the configuration dictionary.

        Args:
            config (dict): Configuration parameters.
        """
        self.num_agents = config["num_agents"]
        self.collision_distance = config["collision_distance"]
        self.sensor_range = config["sensing"]
        self.k = config["k"]
        self.boundary = config["boundary"]
        self.dt = config["dt"]
        self.render_bool = config["render"]
        self.observation_space = {
            "Positions": spaces.Box(low=0, high=self.boundary, shape=(self.num_agents, 2), dtype=np.float32),
            "Distances": spaces.Box(low=0, high=self.sensor_range, shape=(self.num_agents, self.k), dtype=np.float32),
            "Collisions": spaces.Box(low=0, high=1, shape=(self.num_agents, self.k), dtype=np.bool_)
        }        
        self.action_space = list(spaces.Box(low=-1.5, high=1.5, shape=(2,)) for _ in range(self.num_agents))
        self.velocities = np.zeros((self.num_agents, 2))
        self.agents = []
        self.orders = []

        # Additional parameters specific to position initialization and rewards
        self.spacing = config["spacing"]
        self.init_x = config["init_x"]
        self.init_y = config["init_y"]

    def _init_rewards(self, reward_component_strs: List[str], reward_params: Dict):
        """
        Initializes the reward components based on the specified reward component strings and parameters.

        Args:
            reward_component_strs (List[str]): A list of strings representing the names of the reward components to include.
            reward_params (Dict): A dictionary mapping reward component names to their parameters.
        """
        self.reward_initializer = RewardInitializer()
        self.reward_shaper = self.reward_initializer.initialize_reward_shaper(reward_component_strs, reward_params)


    def get_orders(self):
        return self.orders
    
    def _computeReward(self) -> np.ndarray:
        """
        Computes the current reward values for all agents using the reward shaper.

        Returns:
            np.ndarray: An array of shape (num_agents, 1) containing the reward value for each agent.
        """

        # Compute rewards using the reward shaper
        rewards = self.reward_shaper.shape(self.observations)

        return rewards
    
    def _computePeriodicDistances(self) -> None:
        """
        Computes the periodic distances between agents considering the boundary conditions.

        The distances are computed in a way that accounts for the periodic nature of the environment,
        meaning that agents near the edges can still sense and interact with agents on the opposite edge.
        """
        xx1, xx2 = np.meshgrid(self.agents[:, 0], self.agents[:, 0])
        yy1, yy2 = np.meshgrid(self.agents[:, 1], self.agents[:, 1])

        d_ij_x = xx1 - xx2
        d_ij_y = yy1 - yy2
        d_ij_x = np.where(d_ij_x > self.boundary / 2, self.boundary - d_ij_x, d_ij_x)
        d_ij_x = np.where(d_ij_x <= -self.boundary / 2, self.boundary + d_ij_x, d_ij_x)
        
        d_ij_y = np.where(d_ij_y > self.boundary / 2, self.boundary - d_ij_y, d_ij_y)
        d_ij_y = np.where(d_ij_y <= -self.boundary / 2, self.boundary + d_ij_y, d_ij_y)

        self.distances = np.sqrt(np.multiply(d_ij_x, d_ij_x) + np.multiply(d_ij_y, d_ij_y))

        # Nearest neighbors
        distances_to_nearest_neighbors = np.sort(self.distances, axis=1)[:, :self.k + 1]
        nearest_neighbors = np.argsort(-self.distances, axis=1)[:, :self.k + 1]
        
        self.nearest_neighbors = nearest_neighbors[:, 1:]
        self.distances_to_nearest_neighbors = np.clip(distances_to_nearest_neighbors[:, 1:], 0, self.sensor_range)

    def _computeCollisions(self) -> None:
        """
        Computes collisions between agents based on their distances to nearest neighbors.

        A collision is detected if the distance to any nearest neighbor is less than
        the specified collision distance. The result is stored in the `collisions` attribute.
        """
        self.collisions = self.distances_to_nearest_neighbors < self.collision_distance


    def _computeDone(self) -> Tuple[np.ndarray, bool]:
        """
        Computes the done condition for each agent and the overall done condition for the environment.

        The done condition is True if a collision has occurred for any agent.

        Returns:
            Tuple[np.ndarray, bool]: A tuple containing an array of done conditions for each agent and
                                    a boolean indicating if any agent has collided.
        """
        dones = np.any(self.collisions)
        return (self.collisions[:, 0], dones)

    def _computeObs(self) -> dict:
        """
        Computes the observations for the agents.

        The observations include the positions of the agents, their distances to the k nearest neighbors,
        and a boolean array indicating collisions between agents.

        Returns:
            dict: A dictionary containing the positions, distances, and collisions as observations.
        """
        self.observations = {
            "Positions": self.agents,
            "Distances": self.distances_to_nearest_neighbors,
            "Collisions":self.collisions[:, 0]
        }

        # Assert that the observations have the correct dimensions
        assert self.observations["Positions"].shape == (self.num_agents, 3), "Positions observation has incorrect shape."
        assert self.observations["Distances"].shape == (self.num_agents, self.k), f"Distances observation has incorrect shape Now, shape {self.observations['Distances'].shape} != {(self.num_agents, self.k)}."
        assert self.observations["Collisions"].shape == (self.num_agents,), "Collisions observation has incorrect shape."

        return copy(self.observations)

    def _computeKinematics(self, actions: np.ndarray) -> None:
        """
        Updates the positions and headings of the agents based on the given actions, enforcing non-holonomic kinematics.

        Args:
            actions (np.ndarray): An array of shape (num_agents, 2) containing the linear and angular velocities (u, w) for each agent.
        """
        u, w = actions[:, 0], actions[:, 1]

        self.agents[:, 0] += np.multiply(u, np.cos(self.agents[:, 2])) * self.dt
        self.agents[:, 1] += np.multiply(u, np.sin(self.agents[:, 2])) * self.dt
        self.agents[:, 2] = self.wrap_to_pi(self.agents[:, 2] + w * self.dt)
    
    def _checkBoundary(self) -> None:
        """
        Ensures that the agents stay within the boundaries of the environment.

        Agents that cross the boundary are wrapped around to the opposite side.
        """
        self.agents[:, 0] = np.where(
            self.agents[:, 0] < 0, self.boundary, np.where(
                self.agents[:, 0] > self.boundary, 0.001, self.agents[:, 0]
            )
        )

        self.agents[:, 1] = np.where(
            self.agents[:, 1] < 0, self.boundary, np.where(
                self.agents[:, 1] > self.boundary, 0.001, self.agents[:, 1]
            )
        )


    def step(self, actions: np.ndarray) -> Tuple[dict, np.ndarray, np.ndarray, dict]:
        """
        Executes one step in the environment with the given actions.

        Args:
            actions (np.ndarray): An array of shape (num_agents, 2) containing the linear and angular velocities (u, w) for each agent.

        Returns:
            Tuple[dict, np.ndarray, np.ndarray, dict]:
                - observations: A dictionary containing the updated observations for each agent.
                - rewards: An array of shape (num_agents, 1) containing the reward for each agent.
                - dones: A boolean array indicating whether the episode is done for each agent.
                - info: A dictionary containing additional information about the environment.
        """
        self._computeKinematics(actions)
        self._checkBoundary()
        self._computePeriodicDistances()
        self._computeCollisions()
        dones = self._computeDone()
        rewards = self._computeReward()

        self.calculate_order()
        info = {"order": self.orders[-1]}

        observations = self._computeObs()

        return observations, rewards, dones, info

    def reset(self) -> Dict[str, np.ndarray]:
        """
        Resets the environment to its initial state.

        it initializes the positions of the agents using the appropriate method. It then resets various
        state variables such as agents' velocities and orders. Finally, it computes the
        initial distances, collisions, and observations.

        Returns:
            Dict[str, np.ndarray]: The initial observations of the environment.
        """
        pos_xs, pos_ys, pos_hs = self.initialize_positions()
        self.agents = np.column_stack((pos_xs, pos_ys, pos_hs))
        self.orders = []
        self.velocities = np.zeros((self.num_agents, 2))

        self._checkBoundary()
        self._computePeriodicDistances()
        self._computeCollisions()
        _ = self._computeDone()  # 'dones' is not used here, so we ignore it
        observations = self._computeObs()

        return observations
    
    def relative_agent_observations(self):
        observations = {}
        for i in range(self.num_agents):
            relative_positions = np.copy(self.agents)
            relative_positions[:, :2] -= self.agents[i, :2]
            relative_positions[:, 2] = self.wrap_to_pi(relative_positions[:, 2] - self.agents[i, 2])
            observations[i] = relative_positions
        return observations
    
    def render(self):
        """
        Renders the environment visualization.

        The visualization can be exited by pressing the 'q' key.
        """
        self.update_plot(self.agents[:, 0], self.agents[:, 1], self.agents[:, 2])
        plt.gcf().canvas.mpl_connect('key_press_event', self._on_key_press)

    def _on_key_press(self, event):
        """
        Handles key press events in the visualization.

        Args:
            event: The key press event.
        """
        if event.key == 'q':
            plt.close()

    def close(self):
        plt.close()

    def initialize_positions(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Initializes positions for all agents.

        The x and y coordinates are randomly sampled from the entire boundary range.
        The heading angles are randomly sampled from the range [0, 2*pi).

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                Arrays representing the x-coordinates, y-coordinates, and heading angles of the agents.
        """
        pos_xs = np.random.uniform((self.boundary//2) -2, (self.boundary//2)+2, self.num_agents)
        pos_ys = np.random.uniform((self.boundary//2) -2, (self.boundary//2)+2, self.num_agents)
        pos_hs = np.random.uniform(0, 2 * np.pi, self.num_agents)

        return pos_xs, pos_ys, pos_hs
    
    def calculate_order(self):
        pos_hs = self.agents[:, 2]
        cos_sum = np.sum(np.cos(pos_hs))
        sin_sum = np.sum(np.sin(pos_hs))
        order = np.sqrt((cos_sum ** 2) + (sin_sum ** 2)) / pos_hs.shape[0]
        self.orders.append(order)
        

    def plot_order(self):
        time_values = np.arange(len(self.orders)) * self.dt
        plt.figure()
        plt.plot(time_values, self.orders)
        plt.xlabel('Time (s)')
        plt.ylabel('Order')
        plt.title('Swarm Order Over Time')
        plt.show()
        plt.pause(100)



    @staticmethod
    def wrap_to_pi(x):
        x = x % (3.1415926 * 2)
        x = (x + (3.1415926 * 2)) % (3.1415926 * 2)

        x[x > 3.1415926] = x[x > 3.1415926] - (3.1415926 * 2)
        return x
    
    def update_plot(self, pos_xs, pos_ys, pos_hs):
        """
        Updates the plot with the current positions and headings of the agents.

        Args:
            pos_xs (np.ndarray): The x-coordinates of the agents.
            pos_ys (np.ndarray): The y-coordinates of the agents.
            pos_hs (np.ndarray): The headings of the agents.
        """
        plt.scatter(pos_xs, pos_ys, color="black", s=15)
        plt.quiver(pos_xs, pos_ys, np.cos(pos_hs), np.sin(pos_hs), color="black", width=0.005, scale=40)
        plt.axis([0, self.boundary, 0, self.boundary])
        plt.gca().set_facecolor('white')
        plt.draw()
        plt.pause(0.0000001)
        plt.clf()