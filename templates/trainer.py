
from template import *
from dataclasses import dataclass


class Trainer:
    """
    A class for training a PyTorch model
    """
    def __init__(self, model, args=None):

        
        self.model = model



    



@dataclass
class TrainingArguments:

    lr: float = 1e-4
    optimizer: str = "Adam"




