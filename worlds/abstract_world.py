from abc import ABC, abstractmethod


class AbstractWorld(ABC):

    def __init__(self):
        self.agent = None

    def register_agent(self, agent):
        self.agent = agent

    @abstractmethod
    def spawn_agent(self, *args):
        pass

    @abstractmethod
    def agent_is_dead(self):
        # provide situations when agent is dead. Might be removed afterward?
        pass

    @abstractmethod
    def step(self, *args):
        pass
