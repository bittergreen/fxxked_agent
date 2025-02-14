import random

import pygame
import torch

from agent import AgentPosition, VanillaAgent

SCREEN_WIDTH = 200
SCREEN_HEIGHT = 200
BACKGROUND_COLOR = (255, 255, 255)
WORLD_COLOR = (0, 0, 0)


class World:

    def __init__(self, width=SCREEN_WIDTH, height=SCREEN_HEIGHT):
        self.width = width
        self.height = height
        self.agent = None

    def register_agent(self, agent):
        self.agent = agent

    def spawn_agent(self, brain):
        # spawning agent in a random position inside the environment
        self.agent = VanillaAgent(brain=brain)
        init_x = random.randint(0, self.width - self.agent.size)
        init_y = random.randint(0, self.height - self.agent.size)
        self.agent.set_position(AgentPosition(init_x, init_y))

    def spawn_agent_for_training(self, brain):
        """
        For training purposes, in order to generate more cases when the agent go off the environment within reasonable
        length of sequences, we would like to spawn the agent closer to the boarders.

        Note that, interestingly, since here we initialize the agent close to the up-left corner during training,
        we will be able to observe a very strong bias towards the up-left corner from the trained agent.
        That is, if it succeeds in surviving, it will basically always end up looping in circles in the up-left quarter.
        Otherwise, especially if it's spawned somewhere to the bottom-right corner, it might fuck up and die.
        """
        init_x = self.width / 8
        init_y = self.width / 8
        self.agent = VanillaAgent(AgentPosition(init_x, init_y), brain)

    def agent_is_dead(self):
        return self.agent.pos.out_of_boarder(self.width, self.height, self.agent.size)

    def update(self, action):
        self.agent.move(action)


def simulate(brain):
    # for graphical simulation
    pygame.init()

    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("World with Agent")

    world = World(width=SCREEN_WIDTH, height=SCREEN_HEIGHT)
    world.spawn_agent(brain)
    agent = world.agent
    steps = 0

    running = True
    while running and not world.agent_is_dead():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                print("Succeeded in surviving!")
                running = False

        screen.fill(BACKGROUND_COLOR)

        pygame.draw.rect(screen, WORLD_COLOR, pygame.Rect(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT), 5)

        info = agent.pos.to_tensor()
        out = brain(info)
        action = torch.argmax(out, dim=-1)

        world.update(action)

        pygame.draw.rect(screen, agent.color, pygame.Rect(
            agent.pos.x, agent.pos.y, agent.size, agent.size))

        pygame.display.flip()

        pygame.time.Clock().tick(10)

        steps += 1

    print(f"Survived {steps} steps!")

    pygame.quit()

