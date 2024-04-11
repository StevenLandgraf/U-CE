"""
U-CE: Uncertainty-aware Cross-Entropy for Semantic Segmentation
Authors: Steven Landgraf, Markus Hillemann, Kira Wursthorn, Markus Ulrich
"""

from torch.optim.lr_scheduler import _LRScheduler


class PolyLR(_LRScheduler):
    """LR = Initial_LR * (1 - iter / max_iter)^0.9"""
    def __init__(self, optimizer, max_iterations, power=0.9):
        self.current_iteration = 0
        self.max_iterations = max_iterations
        self.power = power
        super().__init__(optimizer)

    def get_lr(self):
        self.current_iteration += 1
        return [base_lr * (1 - self.current_iteration / self.max_iterations) ** self.power for base_lr in self.base_lrs]
