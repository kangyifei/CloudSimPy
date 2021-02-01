from abc import ABC, abstractmethod


class AbstractAgent(ABC):
    @abstractmethod
    def choose_action(self, ob):
        pass

    @abstractmethod
    def learn(self, ob, action, reward):
        pass
