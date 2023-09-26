from abc import ABC, abstractmethod

class Agent(ABC): 

    @abstractmethod
    def __init__(self, cfg):
        self.cfg = cfg

    @abstractmethod
    def save(self):
        pass
    
    @abstractmethod
    def load(self):
        pass
  
    @abstractmethod
    def act(self):
        pass