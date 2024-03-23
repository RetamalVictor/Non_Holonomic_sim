import abc
import numpy as np
from typing import List, Dict


class RewardShaper:
    def __init__(self, reward_components: list, clipping:bool = False, clipping_range:float = 0.9):
        """
        Initialize the reward shaper with a list of reward components.

        Args:
            reward_components (list): A list of reward components to be applied.
            clipping (bool): default False. Clips the reward to a range
            clipping_range (float)
        """
        self.reward_components = reward_components
        self.clipping = clipping
        self.clipping_range = clipping_range

    def shape(self, observation: dict) -> np.ndarray:
        """
        Compute the overall reward based on the observation and reward components.

        Args:
            observation (dict): The observation dictionary containing agent information.

        Returns:
            np.ndarray: An array containing the overall reward for each agent.
        """
        reward = 0
        for component in self.reward_components:
            reward += component.shape(observation)
        if self.clipping:
            return np.clip(reward, -self.clipping_range, self.clipping_range)
        return reward

class RewardComponent(abc.ABC):
    @abc.abstractmethod
    def shape(self, agents, collisions):
        """
        Compute the reward component for each agent based on their current state and interactions.

        Parameters:
            agents (np.ndarray): The array containing the states of the agents.
            collisions (np.ndarray): The array indicating whether each agent has collided.

        Returns:
            np.ndarray: The reward component for each agent.
        """
        pass

class CollisionRewardComponent(RewardComponent):
    def __init__(self, collision_penalty: float, no_collision_reward: float):
        """
        Initialize the collision reward component.

        Args:
            collision_penalty (float): The penalty applied for each collision.
            no_collision_reward (float): The reward given when there is no collision.
        """
        self.collision_penalty = collision_penalty
        self.no_collision_reward = no_collision_reward

    def shape(self, observation: dict) -> np.ndarray:
        """
        Compute the collision penalty based on the collision information in the observation.

        Args:
            observation (dict): The observation dictionary containing collision information.

        Returns:
            np.ndarray: An array containing the collision penalty for each agent.
        """
        collisions = observation['Collisions']
        return np.where(collisions, self.collision_penalty, self.no_collision_reward).reshape(-1, 1)

class SpacingRewardComponent(RewardComponent):
    def __init__(self, desired_distance: float, reward_amount: float):
        """
        Initialize the spacing reward component.

        Args:
            desired_distance (float): The desired distance between agents to maintain.
            reward_amount (float): The reward given for maintaining the desired distance.
        """
        self.desired_distance = desired_distance
        self.reward_amount = reward_amount

    def shape(self, observation: dict) -> np.ndarray:
        """
        Compute the reward for maintaining a desired distance between agents.

        Args:
            observation (dict): The observation dictionary containing agent distances.

        Returns:
            np.ndarray: An array containing the spacing reward for each agent.
        """
        distances = observation['Distances']
        distance_diff = np.abs(distances - self.desired_distance)
        rewards = np.where(distance_diff < self.desired_distance / 2, self.reward_amount, 0)
        return rewards.mean(axis=1).reshape(-1, 1)

class RewardInitializer:
    def __init__(self):
        """
        Initializes the RewardInitializer with a mapping of reward component names to their classes.
        """
        self.component_mapping = {
            "collision": CollisionRewardComponent,
            "spacing": SpacingRewardComponent
            # Add other reward components here as needed
        }

    def initialize_reward_shaper(self, reward_component_strs: List[str], reward_params: Dict) -> RewardShaper:
        """
        Initializes a RewardShaper with the specified reward components and parameters.

        Args:
            reward_component_strs (List[str]): A list of strings representing the names of the reward components to include.
            reward_params (Dict): A dictionary mapping reward component names to their parameters.

        Returns:
            RewardShaper: An instance of RewardShaper with the specified reward components.
        """
        reward_components = []
        for comp_str in reward_component_strs:
            if comp_str in self.component_mapping:
                comp_class = self.component_mapping[comp_str]
                comp_args = reward_params.get(comp_str, {})
                reward_components.append(comp_class(**comp_args))
        return RewardShaper(reward_components)