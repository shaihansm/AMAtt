import torch
from torch.optim import Optimizer
from mAtt import StiefelParameter
from mAtt.utils import orthogonal_projection, retraction

class MixOptimizer:
   
    
    def __init__(self, optimizer: Optimizer) -> None:
        
        self.optimizer = optimizer
        self.state: dict[int, torch.Tensor] = {}
    
    def zero_grad(self) -> None:
        
        self.optimizer.zero_grad()
    
    def step(self, closure: callable = None) -> torch.Tensor:
       
        # Adjust gradients for parameters in the Stiefel space.
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if isinstance(p, StiefelParameter):
                    if id(p) not in self.state:
                        self.state[id(p)] = p.data.clone()
                    else:
                        # Reset and accumulate the current parameter state.
                        self.state[id(p)].fill_(0).add_(p.data)
                    
                    # Clear p.data and update p.grad.data with its orthogonal projection.
                    p.data.fill_(0)
                    trans = orthogonal_projection(p.grad.data, p.data)
                    p.grad.data.fill_(0).add_(trans)
        
        loss = self.optimizer.step(closure)
        
        # Retract the updated Stiefel parameters.
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if isinstance(p, StiefelParameter):
                    trans = retraction(p.data, self.state[id(p)])
                    p.data.fill_(0).add_(trans)
                    
        return loss