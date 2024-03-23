import numpy as np
from reward_component import RewardInitializer

def test_rewards():
    """
    Test function to showcase the functionality of the rewards module with initializer.
    """
    # Dummy data for testing
    agents = np.array([[0, 0], [1, 1], [2, 2]])
    distances = np.sqrt((agents[:, np.newaxis] - agents[np.newaxis, :])**2).sum(axis=-1)
    collisions = np.array([False, True, False])  # Assume the second agent is in a collision

    # Observation dictionary
    observation = {'Positions': agents, 'Distances': distances, 'Collisions': collisions}

    # Specify reward components and their parameters
    reward_strs = ["collision", "spacing"]
    reward_params = {
        "collision": {
            "collision_penalty": -5,
            "no_collision_reward": 0.01
        },
        "spacing": {
            "desired_distance": 1.5,
            "reward_amount": 0.05
        }
    }

    # Initialize the reward initializer and reward shaper
    reward_initializer = RewardInitializer()
    reward_shaper = reward_initializer.initialize_reward_shaper(reward_strs, reward_params)

    # Compute rewards
    rewards = reward_shaper.shape(observation)
    print("Rewards with initializer:", rewards)

if __name__ == "__main__":
    test_rewards()
