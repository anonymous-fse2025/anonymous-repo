import torch


class ScheduledOptim:
    """A simple wrapper class for learning rate scheduling."""

    def __init__(self, optimizer, d_model, n_warmup_steps):
        self._optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        self.init_lr = d_model ** -0.5

    def step_and_update_lr(self):
        """Perform a step with the inner optimizer and update the learning rate."""
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        """Zero out the gradients by the inner optimizer."""
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        """Calculate the learning rate scale factor."""
        return min(self.n_current_steps ** -0.5, self.n_current_steps * (self.n_warmup_steps ** -1.5))

    def _update_learning_rate(self):
        """Update learning rate at each step based on warmup strategy."""
        self.n_current_steps += 1
        lr = self.init_lr * self._get_lr_scale()
        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr
