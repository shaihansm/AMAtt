import torch
import torch.nn as nn
from mAtt.spd import SPDTransform, SPDTangentSpace, SPDRectified

class signal2spd(nn.Module):
   
    def __init__(self) -> None:
        super().__init__()
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Remove singleton dimensions if any.
        x = x.squeeze()
        mean = x.mean(dim=-1).unsqueeze(-1).repeat(1, 1, x.shape[-1])
        x = x - mean
        # Compute covariance-like matrix.
        cov = x @ x.transpose(1, 2)
        cov = cov.to(self.dev)
        cov = cov / (x.shape[-1] - 1)
        tra = cov.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)
        tra = tra.view(-1, 1, 1)
        cov = cov / tra
        identity = torch.eye(cov.shape[-1], device=self.dev).repeat(x.shape[0], 1, 1)
        cov = cov + 1e-5 * identity
        return cov 

class E2R(nn.Module):
    
    # Converts each epoch (time-series) of the EEG signal into an SPD matrix by splitting the time dimension. Implements Adaptive Log-Euclidean Metrics (ALEM) for SPD matrix learning.
    
    def __init__(self, epochs: int) -> None:
        
        super().__init__()
        self.epochs = epochs
        self.signal2spd = signal2spd()

        # Learnable parameters for adaptive scaling and shifting
        self.alpha = nn.Parameter(torch.ones(1, 1, 1)*0.1)  # Shape: [C, 1, 1]
        self.beta = nn.Parameter(torch.zeros(1, 1, 1)*0.1)   # Shape: [C, 1, 1]

    def tensor_log(self, t: torch.Tensor) -> torch.Tensor:
        s, u = torch.linalg.eigh(t)
        s = torch.clamp(s, min=1e-6)  # Prevents log instability
        return u @ torch.diag_embed(torch.log(s)) @ u.transpose(-2, -1)


    def tensor_exp(self, t: torch.Tensor) -> torch.Tensor:
    
        # Symmetrize input to ensure valid SPD output
        t_sym = (t + t.transpose(-2, -1)) / 2

        # Compute eigendecomposition
        s, u = torch.linalg.eigh(t_sym)

        # Clamp eigenvalues to prevent exploding gradients
        s = torch.clamp(s, min=-50, max=50)

        # Reconstruct exp(t)
        return u @ torch.diag_embed(torch.exp(s)) @ u.transpose(-2, -1)

    def patch_len(self, n: int, epochs: int) -> list:
 
        list_len = []
        base = n // epochs
        for i in range(epochs):
            list_len.append(base)
        for i in range(n - base * epochs):
            list_len[i] += 1

        if sum(list_len) == n:
            return list_len
        else:
            raise ValueError("Check your epochs and axis splitting.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
      
        # Split EEG signal into epochs and convert to SPD matrices
        list_patch = self.patch_len(x.shape[-1], int(self.epochs))
        x_list = list(torch.split(x, list_patch, dim=-1))
        for i, item in enumerate(x_list):
            x_list[i] = self.signal2spd(item)
        x = torch.stack(x_list).permute(1, 0, 2, 3)  # Shape: [B, T, C, C]

        # Apply Adaptive Log-Euclidean transformation
        log_x = self.tensor_log(x)  # Shape: [B, T, C, C]
        adapted_log_x = self.alpha * log_x + self.beta
        adapted_x = self.tensor_exp(adapted_log_x)
        
        return adapted_x

class AttentionManifold(nn.Module):
    
    def __init__(self, in_embed_size: int, out_embed_size: int) -> None:
        super().__init__()
        self.d_in = in_embed_size
        self.d_out = out_embed_size
        self.q_trans = SPDTransform(self.d_in, self.d_out).cpu()
        self.k_trans = SPDTransform(self.d_in, self.d_out).cpu()
        self.v_trans = SPDTransform(self.d_in, self.d_out).cpu()

    def tensor_log(self, t: torch.Tensor) -> torch.Tensor:
        # Use torch.linalg.svd; t is assumed to be 4D: [batch, *, s, s]
        U, S, Vh = torch.linalg.svd(t, full_matrices=False)
        V = Vh.transpose(-2, -1)
        return U @ torch.diag_embed(torch.log(S)) @ V
        
    def tensor_exp(self, t: torch.Tensor) -> torch.Tensor:
        # t is assumed to be symmetric.
        s, u = torch.linalg.eigh(t)
        return u @ torch.diag_embed(torch.exp(s)) @ u.transpose(-2, -1)
    
    def log_euclidean_distance(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        inner_term = self.tensor_log(A) - self.tensor_log(B)
        inner_multi = inner_term @ inner_term.transpose(-2, -1)
        _, s, _ = torch.linalg.svd(inner_multi, full_matrices=False)
        final = torch.sum(s, dim=-1)
        return final

    def LogEuclideanMean(self, weight: torch.Tensor, cov: torch.Tensor) -> torch.Tensor:
        # cov: [bs, num_p, s, s]; weight: [bs, num_p, num_p]
        bs, num_p, s_size, _ = cov.shape
        cov_log = self.tensor_log(cov).view(bs, num_p, -1)
        output = weight @ cov_log  # [bs, num_p, -1]
        output = output.view(bs, num_p, s_size, s_size)
        return self.tensor_exp(output)
        
    def forward(self, x: torch.Tensor, shape: list = None) -> tuple:
        # If a 3D input is given with a provided shape, reshape x accordingly.
        if len(x.shape) == 3 and shape is not None:
            x = x.view(shape[0], shape[1], self.d_in, self.d_in)
        x = x.to(torch.float)
        bs, m = x.shape[0], x.shape[1]
        x = x.reshape(bs * m, self.d_in, self.d_in)
        Q = self.q_trans(x).view(bs, m, self.d_out, self.d_out)
        K = self.k_trans(x).view(bs, m, self.d_out, self.d_out)
        V = self.v_trans(x).view(bs, m, self.d_out, self.d_out)

        Q_expand = Q.repeat(1, V.shape[1], 1, 1)
        K_expand = K.unsqueeze(2).repeat(1, 1, V.shape[1], 1, 1)
        K_expand = K_expand.view(bs, K.shape[1] * V.shape[1], self.d_out, self.d_out)
        
        atten_energy = self.log_euclidean_distance(Q_expand, K_expand).view(bs, V.shape[1], V.shape[1])
        atten_prob = nn.Softmax(dim=-2)(1 / (1 + torch.log(1 + atten_energy))).permute(0, 2, 1)
        
        output = self.LogEuclideanMean(atten_prob, V)
        output = output.view(bs, V.shape[1], self.d_out, self.d_out)
        shape_out = [output.shape[0], output.shape[1], -1]
        output = output.contiguous().view(-1, self.d_out, self.d_out)
        return output, shape_out

class mAtt_bci(nn.Module):
    
    def __init__(self, epochs: int) -> None:
        super().__init__()
        # Feature Extraction
        self.conv1 = nn.Conv2d(1, 22, (22, 1))
        self.Bn1 = nn.BatchNorm2d(22)
        self.conv2 = nn.Conv2d(22, 20, (1, 12), padding=(0, 6))
        self.Bn2 = nn.BatchNorm2d(20)
        
        # Euclidean-to-Riemannian conversion
        self.ract1 = E2R(epochs=epochs)
        # Riemannian attention and rectification
        self.att2 = AttentionManifold(20, 18)
        self.ract2 = SPDRectified()
        # Riemannian-to-Euclidean conversion
        self.tangent = SPDTangentSpace(18)
        self.flat = nn.Flatten()
        # Final classification
        self.linear = nn.Linear(9 * 19 * epochs, 4, bias=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.Bn1(x)
        x = self.conv2(x)
        x = self.Bn2(x)
        
        x = self.ract1(x)
        x, shape = self.att2(x)
        x = self.ract2(x)
        x = self.tangent(x)
        x = x.view(shape[0], shape[1], -1)
        x = self.flat(x)
        x = self.linear(x)
        return x

class mAtt_mamem(nn.Module):
    
    def __init__(self, epochs: int) -> None:
        super().__init__()
        # Feature Extraction
        self.conv1 = nn.Conv2d(1, 125, (8, 1))
        self.Bn1 = nn.BatchNorm2d(125)
        self.conv2 = nn.Conv2d(125, 15, (1, 36), padding=(0, 18))
        self.Bn2 = nn.BatchNorm2d(15)
        
        # Euclidean-to-Riemannian conversion
        self.ract1 = E2R(epochs)
        # Riemannian attention and rectification
        self.att2 = AttentionManifold(15, 12)
        self.ract2 = SPDRectified()
        # Riemannian-to-Euclidean conversion
        self.tangent = SPDTangentSpace(12)
        self.flat = nn.Flatten()
        # Final classifier
        self.linear = nn.Linear(6 * 13 * epochs, 5, bias=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.Bn1(x)
        x = self.conv2(x)
        x = self.Bn2(x)
        
        x = self.ract1(x)
        x, shape = self.att2(x)
        x = self.ract2(x)
        x = self.tangent(x)
        x = x.view(shape[0], shape[1], -1)
        x = self.flat(x)
        x = self.linear(x)
        return x
