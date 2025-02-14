from environments.vanilla_env import simulate
from train import train_vanilla_agent


if __name__ == '__main__':
    model = train_vanilla_agent()
    num_simulations = 10
    for _ in range(num_simulations):
        simulate(model)

