import torch
from torch import nn
from torch.autograd import Function
import numpy as np
from mAtt.utils import symmetric  # Assumes a 'symmetric' function exists in mAtt/utils
from mAtt import StiefelParameter

class SPDTransform(nn.Module):
    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.increase_dim = None
        if output_size > input_size:
            self.increase_dim = SPDIncreaseDim(input_size, output_size)
            input_size = output_size
        # Create a StiefelParameter with proper initialization on the device.
        self.weight = StiefelParameter(torch.empty(input_size, output_size, device=self.device), requires_grad=True)
        nn.init.orthogonal_(self.weight)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = input
        if self.increase_dim is not None:
            output = self.increase_dim(output)
        weight = self.weight.unsqueeze(0).expand(input.size(0), -1, -1)
        output = torch.bmm(weight.transpose(1, 2), torch.bmm(output, weight))
        return output


class SPDIncreaseDim(nn.Module):
    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.register_buffer('eye', torch.eye(output_size, input_size, device=self.device))
        add = torch.tensor([0] * input_size + [1] * (output_size - input_size),
                           dtype=torch.float32, device=self.device)
        self.register_buffer('add', torch.diag(add))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        eye = self.eye.unsqueeze(0).expand(input.size(0), -1, -1)
        add = self.add.unsqueeze(0).expand(input.size(0), -1, -1)
        output = torch.baddbmm(add, eye, torch.bmm(input, eye.transpose(1, 2)))
        return output


class ParametricVectorize(nn.Module):
    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.weight = nn.Parameter(torch.ones(output_size, input_size, device=self.device), requires_grad=True)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = self.weight.unsqueeze(0).expand(input.size(0), -1, -1)
        output = torch.bmm(weight, input)
        output = torch.bmm(output, weight.transpose(1, 2))
        output = torch.mean(output, dim=2)
        return output


class SPDVectorize(nn.Module):
    def __init__(self, input_size: int) -> None:
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        row_idx, col_idx = torch.triu_indices(input_size, input_size)
        self.register_buffer('row_idx', row_idx.to(self.device))
        self.register_buffer('col_idx', col_idx.to(self.device))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = input[:, self.row_idx, self.col_idx]
        return output


class SPDUnVectorizeFunction(Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(input)
        n = int(-0.5 + 0.5 * torch.sqrt(1 + 8 * input.size(1)))
        output = input.new_zeros(len(input), n, n)
        # Create masks using current device
        mask_upper = torch.triu_indices(n, n, device=input.device)
        for k, x in enumerate(input):
            # Fill upper-triangular part.
            output[k].view(-1)[mask_upper[0] * n + mask_upper[1]] = x
            output[k] = output[k] + output[k].t()
            diag_idx = torch.arange(n, device=input.device)
            output[k][diag_idx, diag_idx] /= 2
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        input, = ctx.saved_tensors
        grad_input = None
        if ctx.needs_input_grad[0]:
            n = int(-0.5 + 0.5 * torch.sqrt(1 + 8 * input.size(1)))
            grad_input = input.new_zeros(len(input), input.size(1))
            mask = torch.triu_indices(n, n, device=input.device)
            for k, g in enumerate(grad_output):
                grad_input[k] = g[mask[0] * n + mask[1]]
        return grad_input

class SPDUnVectorize(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return SPDUnVectorizeFunction.apply(input)


class SPDTangentSpaceFunction(Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(input)
        output = input.new_empty(input.size(0), input.size(1), input.size(2))
        for k, x in enumerate(input):
            # Use torch.linalg.svd for a numerically stable decomposition.
            u, s, vh = torch.linalg.svd(x, full_matrices=False)
            s = torch.log(s)
            s_diag = torch.diag_embed(s)
            output[k] = u @ s_diag @ u.transpose(-2, -1)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        input, = ctx.saved_tensors
        grad_input = None
        if ctx.needs_input_grad[0]:
            device = input.device
            eye = torch.eye(input.size(1), device=device)
            grad_input = input.new_empty(input.size(0), input.size(1), input.size(1))
            for k, g in enumerate(grad_output):
                x = input[k]
                u, s, vh = torch.linalg.svd(x, full_matrices=False)
                g_sym = symmetric(g)
                s_log_diag = torch.diag_embed(torch.log(s))
                s_inv_diag = torch.diag_embed(1 / s)
                dLdV = 2 * (g_sym @ (u @ s_log_diag))
                dLdS = eye * (s_inv_diag @ (u.transpose(0, 1) @ (g_sym @ u)))
                P = s.unsqueeze(1) - s.unsqueeze(0)
                mask_zero = (P.abs() < 1e-8)
                P_inv = torch.where(mask_zero, torch.zeros_like(P), 1 / P)
                grad_input[k] = u @ (symmetric(P_inv.transpose(0, 1) * (u.transpose(0, 1) @ dLdV)) + dLdS) @ u.transpose(0, 1)
        return grad_input

class SPDTangentSpace(nn.Module):
    def __init__(self, input_size: int, vectorize: bool = True) -> None:
        super().__init__()
        self.vectorize = vectorize
        if vectorize:
            self.vec = SPDVectorize(input_size)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = SPDTangentSpaceFunction.apply(input)
        if self.vectorize:
            output = self.vec(output)
        return output


class SPDUnTangentSpaceFunction(Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(input)
        output = input.new_empty(input.size(0), input.size(1), input.size(2))
        for k, x in enumerate(input):
            u, s, vh = torch.linalg.svd(x, full_matrices=False)
            s = torch.exp(s)
            s_diag = torch.diag_embed(s)
            output[k] = u @ s_diag @ u.transpose(-2, -1)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        input, = ctx.saved_tensors
        grad_input = None
        if ctx.needs_input_grad[0]:
            device = input.device
            eye = torch.eye(input.size(1), device=device)
            grad_input = input.new_empty(input.size(0), input.size(1), input.size(1))
            for k, g in enumerate(grad_output):
                x = input[k]
                u, s, vh = torch.linalg.svd(x, full_matrices=False)
                g_sym = symmetric(g)
                s_exp_diag = torch.diag_embed(torch.exp(s))
                dLdV = 2 * (g_sym @ (u @ s_exp_diag))
                dLdS = eye * (s_exp_diag @ (u.transpose(0, 1) @ (g_sym @ u)))
                P = s.unsqueeze(1) - s.unsqueeze(0)
                mask_zero = (P.abs() < 1e-8)
                P_inv = torch.where(mask_zero, torch.zeros_like(P), 1 / P)
                grad_input[k] = u @ (symmetric(P_inv.transpose(0, 1) * (u.transpose(0, 1) @ dLdV)) + dLdS) @ u.transpose(0, 1)
        return grad_input

class SPDUnTangentSpace(nn.Module):
    def __init__(self, unvectorize: bool = True) -> None:
        super().__init__()
        self.unvectorize = unvectorize
        if unvectorize:
            self.unvec = SPDUnVectorize()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.unvectorize:
            input = self.unvec(input)
        output = SPDUnTangentSpaceFunction.apply(input)
        return output


class SPDRectifiedFunction(Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, epsilon: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(input, epsilon)
        output = input.new_empty(input.size(0), input.size(1), input.size(2))
        for k, x in enumerate(input):
            u, s, vh = torch.linalg.svd(x, full_matrices=False)
            s = torch.where(s < epsilon[0], epsilon[0], s)
            s_diag = torch.diag_embed(s)
            output[k] = u @ s_diag @ u.transpose(-2, -1)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        input, epsilon = ctx.saved_tensors
        grad_input = None
        if ctx.needs_input_grad[0]:
            device = input.device
            eye = torch.eye(input.size(1), device=device)
            grad_input = input.new_empty(input.size(0), input.size(1), input.size(2))
            for k, g in enumerate(grad_output):
                if g.dim() == 1:
                    continue
                g_sym = symmetric(g)
                x = input[k]
                u, s, vh = torch.linalg.svd(x, full_matrices=False)
                max_mask = s > epsilon
                s_max = s.clone()
                s_max[~max_mask] = epsilon[0]
                s_max_diag = torch.diag_embed(s_max)
                Q = torch.diag_embed(max_mask.float())
                dLdV = 2 * (g_sym @ (u @ s_max_diag))
                dLdS = eye * (Q @ (u.transpose(0, 1) @ (g_sym @ u)))
                P = s.unsqueeze(1) - s.unsqueeze(0)
                mask_zero = (P.abs() < 1e-8)
                P_inv = torch.where(mask_zero, torch.zeros_like(P), 1 / P)
                grad_input[k] = u @ (symmetric(P_inv.transpose(0, 1) * (u.transpose(0, 1) @ dLdV)) + dLdS) @ u.transpose(0, 1)
        return grad_input, None

class SPDRectified(nn.Module):
    def __init__(self, epsilon: float = 1e-5) -> None:
        super().__init__()
        self.epsilon = torch.tensor([epsilon], dtype=torch.float32)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return SPDRectifiedFunction.apply(input, self.epsilon)