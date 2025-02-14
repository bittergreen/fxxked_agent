import torch

from brain import Brain


AGENT_SIZE = 20
AGENT_COLOR = (255, 0, 0)
AGENT_SPEED = 20


class AgentPosition:

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def move_up(self, speed):
        self.y -= speed
        return self

    def move_down(self, speed):
        self.y += speed
        return self

    def move_left(self, speed):
        self.x -= speed
        return self

    def move_right(self, speed):
        self.x += speed
        return self

    def out_of_boarder(self, world_width, world_height, size):
        left = self.x
        up = self.y
        right = self.x + size
        down = self.y + size
        return not (left >= 0 and right <= world_width and up >= 0 and down <= world_height)

    def to_tensor(self):
        # 3-D tensor of (batch, sequence_length, data). Here sequence_length is just 1.
        return torch.tensor([[[self.x, self.y]]], dtype=torch.float)

    def show(self):
        # for debugging
        print(f"{self.x}, {self.y}")

    def copy(self):
        return AgentPosition(self.x, self.y)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y


class Agent:

    def __init__(self, position: AgentPosition = None, brain: Brain = None):
        self.pos = position
        self.brain = brain

    def set_position(self, position: AgentPosition):
        self.pos = position

    def set_brain(self, brain: Brain):
        self.brain = brain


class VanillaAgent(Agent):

    """
    An agent that moves freely in the world, has its size
    """
    def __init__(self, position: AgentPosition = None, brain: Brain = None):
        super().__init__(position, brain)
        self.size = AGENT_SIZE
        self.color = AGENT_COLOR
        self.speed = AGENT_SPEED

    def move(self, action):
        """
        action:
        0 - up
        1 - down
        2 - left
        3 - right
        """
        if action == 0:
            self.pos.move_up(self.speed)
        elif action == 1:
            self.pos.move_down(self.speed)
        elif action == 2:
            self.pos.move_left(self.speed)
        elif action == 3:
            self.pos.move_right(self.speed)
        else:
            raise RuntimeError(f"Invalid action: {action}!")

    def gen_possible_next_positions(self):
        # currently not used
        return [self.pos.copy().move_up(self.speed),
                self.pos.copy().move_down(self.speed),
                self.pos.copy().move_left(self.speed),
                self.pos.copy().move_right(self.speed)]
