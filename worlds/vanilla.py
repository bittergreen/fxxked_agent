import random

import pygame
import torch

from brain import Brain
from worlds.abstract_world import AbstractWorld
from agent import AgentPosition, VanillaAgent

SCREEN_WIDTH = 200
SCREEN_HEIGHT = 200
BACKGROUND_COLOR = (255, 255, 255)
WORLD_COLOR = (0, 0, 0)


class VanillaWorld(AbstractWorld):

    """
    A simple rectangle world.
    """

    def __init__(self, width=SCREEN_WIDTH, height=SCREEN_HEIGHT):
        super().__init__()
        self.width = width
        self.height = height

    def spawn_agent(self, brain):
        # spawning agent in a random position inside the environment
        self.agent = VanillaAgent(brain=brain)
        init_x = random.randint(0, self.width - self.agent.size)
        init_y = random.randint(0, self.height - self.agent.size)
        self.agent.set_position(AgentPosition(init_x, init_y))

    def agent_is_dead(self):
        pos = self.agent.pos
        left = pos.x
        up = pos.y
        right = pos.x + self.agent.size
        down = pos.y + self.agent.size
        return not (left >= 0 and right <= self.width and up >= 0 and down <= self.height)

    def step(self, action):
        self.agent.move(action)

    def simulate(self, brain: Brain):
        pygame.init()

        screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("World with Agent")
        self.spawn_agent(brain)
        steps = 0

        running = True
        while running and not self.agent_is_dead():
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print("Succeeded in surviving!")
                    running = False

            screen.fill(BACKGROUND_COLOR)

            pygame.draw.rect(screen, WORLD_COLOR, pygame.Rect(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT), 5)

            info = self.agent.pos.to_tensor()
            out = brain(info)
            action = torch.argmax(out, dim=-1)

            self.step(action)

            pygame.draw.rect(screen, self.agent.color, self.agent.to_rectangle())

            pygame.display.flip()

            pygame.time.Clock().tick(10)

            steps += 1

        print(f"Survived {steps} steps!")

        pygame.quit()
