import torch
from torch import optim

from agent import AgentPosition
from brain import SurvivalLSTM
from worlds.vanilla import VanillaWorld


def train_vanilla_agent():

    model = SurvivalLSTM()
    world = VanillaWorld()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 1000

    prev_positions = []
    max_step = 30
    # Forcing model not to repeat the recent #memory_length steps of movements
    # Larger memory_length might force the agent to draw more complex patterns,
    # while it might need longer max_step and more num_epochs to train
    memory_length = 3

    def compute_reward(position: AgentPosition):
        reward = 1.0
        if world.agent_is_dead():
            reward = -10.0
        if position in prev_positions:
            reward = -2.0
        return reward

    def train_step():
        model.reset_hidden_state()
        for n in range(max_step):
            pos = agent.pos.to_tensor()  # current position is maintained by agent.pos
            pos_copy = agent.pos.copy()
            prev_positions.append(pos_copy)  # recording visited position within #memory_length steps
            out = model(pos)  # a softmax output of the probability distribution of the 4 possible actions

            action = torch.argmax(out, dim=-1).item()
            log_prob = torch.log(out[:, action])  # compute the log probability of the chosen action
            world.step(action)

            new_pos = agent.pos
            reward = compute_reward(new_pos)
            loss = -log_prob * reward  # policy gradient update for reinforcement learning

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if len(prev_positions) > memory_length:
                prev_positions.pop(0)

            if world.agent_is_dead():
                print(f"Max step: {n}")
                break

    # training
    for i in range(num_epochs):
        print(f"Epoch num: {i}")
        # need to reset agent position every single epoch
        world.spawn_agent(model)
        agent = world.agent
        train_step()

    return world, model

