from train import train_vanilla_agent


if __name__ == '__main__':
    world, model = train_vanilla_agent()
    num_simulations = 10
    for _ in range(num_simulations):
        world.simulate(model)

