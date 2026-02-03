import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
import random


class GMD:
    def __init__(self, optimizer, reduction='mean'):
        """
        optimizer: torch.optim.Optimizer
        reduction: str, 'mean' or 'sum'
        """
        self._optim, self._reduction = optimizer, reduction

    @property
    def optimizer(self):
        return self._optim

    def zero_grad(self):
        return self._optim.zero_grad(set_to_none=True)

    def step(self):
        return self._optim.step()

    def pc_backward(self, objectives):
        """
        Calculate gradients of objectives and project conflicting components.
        objectives: list of torch.Tensor
        """
        grads, shapes, has_grads = self._pack_grad(objectives)
        pc_grad = self._project_conflicting(grads, has_grads)
        pc_grad = self._unflatten_grad(pc_grad, shapes[0])
        self._set_grad(pc_grad)
        return

    def _project_conflicting(self, grads, has_grads, shape=None):
        # Determine shared parameters (parameters that have gradients in all tasks)
        shared = torch.stack(has_grads).prod(0).bool()
        pc_grad, num_stack = copy.deepcopy(grads), len(grads)

        # Iterate over each task's gradient g_i
        for g_i in pc_grad:
            indices = list(range(num_stack))
            random.shuffle(indices)

            for index in indices:
                g_j = grads[index]
                g_i_g_j = torch.dot(g_i, g_j)
                if g_i_g_j < 0:
                    coef = g_i_g_j / (g_j.norm() ** 2)
                    g_i -= coef * g_j

        merged_grad = torch.zeros_like(grads[0]).to(grads[0].device)

        if self._reduction == 'mean':
            merged_grad[shared] = torch.stack([g[shared] for g in pc_grad]).mean(dim=0)
        elif self._reduction == 'sum':
            merged_grad[shared] = torch.stack([g[shared] for g in pc_grad]).sum(dim=0)
        else:
            raise ValueError('invalid reduction method')

        # For non-shared parameters, simply sum them up
        merged_grad[~shared] = torch.stack([g[~shared] for g in pc_grad]).sum(dim=0)
        return merged_grad

    def _set_grad(self, grads):
        # Assign the modified gradients back to network parameters
        idx = 0
        for group in self.optimizer.param_groups:
            for p in group['params']:
                # Skip if original grad was None (to keep index aligned)
                # None: Logic assumes _retrieve_grad fills None with 0s, so we just assign
                p.grad = grads[idx]
                idx += 1
        return

    def _pack_grad(self, objectives):
        """
        Compute and flatten gradients for each objective separately.
        Returns:
            grads: list of flattened gradient tensors
            shapes: list of parameter shapes
            has_grads: list of masks (1 if param has grad, 0 if None)
        """
        grads, shapes, has_grads = [], [], []
        for ii, obj in enumerate(objectives):
            self._optim.zero_grad(set_to_none=True)

            # Retain graph for all but the last objective
            retain_graph = ii < len(objectives) - 1
            obj.backward(retain_graph=retain_graph)

            grad, shape, has_grad = self._retrieve_grad()
            grads.append(self._flatten_grad(grad, shape))
            has_grads.append(self._flatten_grad(has_grad, shape))
            shapes.append(shape)
        return grads, shapes, has_grads

    def _retrieve_grad(self):
        """
        Retrieve gradients from parameters.
        Handles None gradients by creating zero tensors and updating masks.
        """
        grad, shape, has_grad = [], [], []
        for group in self._optim.param_groups:
            for p in group['params']:
                if p.grad is None:
                    shape.append(p.shape)
                    grad.append(torch.zeros_like(p).to(p.device))
                    has_grad.append(torch.zeros_like(p).to(p.device))
                    continue
                shape.append(p.grad.shape)
                grad.append(p.grad.clone())
                has_grad.append(torch.ones_like(p).to(p.device))
        return grad, shape, has_grad

    def _flatten_grad(self, grads, shapes):
        return torch.cat([g.flatten() for g in grads])

    def _unflatten_grad(self, grads, shapes):
        unflatten_grad, idx = [], 0
        for shape in shapes:
            length = np.prod(shape)
            unflatten_grad.append(grads[idx:idx + length].view(shape).clone())
            idx += length
        return unflatten_grad