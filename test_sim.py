import numpy as np
import matplotlib.pyplot as plt
from simulation.simulation import ActiveElasticFlockingEnv
from controllers.flocking import Controller

def test_simulation():
    # Path to the configuration file
    config_path = "config\config.json"

    # Initialize the environment
    env = ActiveElasticFlockingEnv(config_path)
    control = Controller(config_path)

    # Number of steps to simulate
    num_steps = 10000
    observations = env.reset()
    # Simulate the environment
    for step in range(num_steps):
        # Generate random actions for each agent
        # actions = np.random.uniform(0.5, 1.5, (env.num_agents, 2))
        actions = control.calculate_control(observations=observations)
        # Step the environment
        observations, rewards, dones, info = env.step(actions.T)
        print(actions.T[1])
        # Render the environment (if enabled)
        if env.render_bool:
            env.render()

        # Check if the episode is done
        if dones[1]:
            print(f"Episode ended at step {step}")
            # break

    # Close the environment
    env.close()

if __name__ == "__main__":
    test_simulation()
