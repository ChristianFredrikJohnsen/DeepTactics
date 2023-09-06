from abc import ABC, abstractmethod

# Kan inherite fra denne klassen!
# Kan derimot ikke instantiate!



class Agent(ABC): 

    @abstractmethod
    def __init__(self, cfg):
        self.cfg = cfg

    
    def save(self):
        raise NotImplemented
    
    def load(self):
        raise NotImplemented


    # Tar inn state og gjør actions basert på states.    
    def act(self):
        raise NotImplemented
    




































