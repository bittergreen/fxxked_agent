import torch
from torch import optim

from agent import AgentPosition
from brain import SurvivalLSTM
from environments.vanilla_env import World


def train_vanilla_agent():

    model = SurvivalLSTM()
    world = World()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 1000

    prev_positions = []
    max_step = 30
    # Forcing model not to repeat the recent #memory_length steps of movements
    # Larger memory_length might force the agent to draw more complex patterns,
    # while it might need longer max_step and more num_epochs to train
    memory_length = 5

    def compute_reward(position: AgentPosition):
        # return a single value of reward, ranging within [-1, 1]
        reward = 1.0
        if position.out_of_boarder(world.width, world.height, agent.size):
            reward = -10.0
        if position in prev_positions:
            reward = -2.0
        return reward

    def train_step():
        for n in range(max_step):
            pos = agent.pos.to_tensor()  # current position is maintained by agent.pos
            pos_copy = agent.pos.copy()
            prev_positions.append(pos_copy)  # recording visited position within #memory_length steps
            out = model(pos)  # a softmax output of the probability distribution of the 4 possible actions

            action = torch.argmax(out, dim=-1).item()
            log_prob = torch.log(out[:, action])  # compute the log probability of the chosen action
            agent.move(action)
            new_pos = agent.pos
            reward = compute_reward(new_pos)
            loss = -log_prob * reward  # policy gradient update for reinforcement learning

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if len(prev_positions) > memory_length:
                prev_positions.pop(0)

            if new_pos.out_of_boarder(world.width, world.height, agent.size):
                print(f"Max step: {n}")
                break

    # training
    for i in range(num_epochs):
        print(f"Epoch num: {i}")
        # need to reset agent position every single epoch
        world.spawn_agent_for_training(model)
        agent = world.agent
        train_step()

    return model

