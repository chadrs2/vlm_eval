"""
Parts of this code are adapted from the ViT implementation in the timm library
and the NAClip model implementation: https://github.com/sinahmr/NACLIP.
"""

from timm.models.vision_transformer import Attention
import torch
import torch.nn.functional as F
import math

class NAClipAttention(Attention):
    """
    A custom attention class that inherits from timm's Attention.
    This class is designed to handle the specific requirements of the NAClip model.
    """
    def __init__(self, orig_attn, addition_cache: dict, num_summary_tokens: int, num_patches: tuple, attn_strategy: str = 'naclip', gaussian_std: float = 5.0):
        """
        Initialize the NAClipAttention with the original attention parameters.
        
        Args:
            orig_attn: The original attention layer from which to inherit parameters.
        """
        super(Attention, self).__init__()

        self.num_heads = orig_attn.num_heads
        self.head_dim = orig_attn.head_dim
        self.scale = orig_attn.scale
        self.fused_attn = orig_attn.fused_attn

        self.qkv = orig_attn.qkv
        self.q_norm = orig_attn.q_norm
        self.k_norm = orig_attn.k_norm
        self.attn_drop = orig_attn.attn_drop
        self.proj = orig_attn.proj
        self.proj_drop = orig_attn.proj_drop

        self.attn_strategy = attn_strategy
        self.gaussian_std = gaussian_std
        self.num_summary_tokens = num_summary_tokens
        self.n_patches = num_patches
        self.addition_cache = addition_cache

        self.supported_strategies = ['naclip', 'nonly', 'kkonly']
        if self.attn_strategy not in self.supported_strategies:
            raise ValueError(f"{self.attn_strategy} is not supported. Choose from {self.supported_strategies}")

    @staticmethod
    def gaussian_window(dim1, dim2, std=1.):
        constant = 1 / (std * math.sqrt(2))
        ks = list()
        for dim in [dim1, dim2]:
            start = -(dim - 1) / 2.0
            k = torch.linspace(start=start * constant,
                               end=(start + (dim - 1)) * constant,
                               steps=dim,
                               dtype=torch.float)
            ks.append(k)
        dist_square_to_mu = (torch.stack(torch.meshgrid(*ks, indexing='ij')) ** 2).sum(0)
        return torch.exp(-dist_square_to_mu)

    @staticmethod
    def get_attention_addition(dim1, dim2, window, num_cls=None, adjust_for_cls=True):
        m = torch.einsum('ij,kl->ijkl', torch.eye(dim1), torch.eye(dim2))
        m = m.permute((0, 3, 1, 2)).contiguous()  # m[ijkl] = 1 iff (i, j) == (k, l)
        out = F.conv2d(m.view(-1, dim1, dim2).unsqueeze(1), window.unsqueeze(0).unsqueeze(1), padding='same').squeeze(1)
        out = out.view(dim1 * dim2, dim1 * dim2)
        if adjust_for_cls:
            if num_cls is None:
                num_cls = 1
            v_adjusted = torch.vstack([torch.zeros((num_cls, dim1 * dim2)), out])
            out = torch.hstack([torch.zeros((dim1 * dim2 + num_cls, num_cls)), v_adjusted])
        return out

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None, is_causal: bool = False) -> torch.Tensor:
        """
        Forward pass for the NAClipAttention.
        """
        
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn and self.attn_strategy == 'kkonly':
            x = F.scaled_dot_product_attention(
                k, k, v,
                attn_mask=attn_mask,
                is_causal=is_causal,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale

            if self.attn_strategy == 'naclip' or self.attn_strategy == 'kkonly':
                attn = k @ k.transpose(-2, -1)
            elif self.attn_strategy == 'nonly':
                attn = q @ k.transpose(-2, -1)
            else:
                raise ValueError(f"{self.attn_strategy} is not supported. Choose from {self.supported_strategies}")

            if self.attn_strategy == 'naclip' or self.attn_strategy == 'nonly':
                addition = self.addition_cache.get(self.n_patches)
                if addition is None:
                    window_size = [side * 2 - 1 for side in self.n_patches]
                    window = NAClipAttention.gaussian_window(*window_size, std=self.gaussian_std)
                    addition = NAClipAttention.get_attention_addition(
                        *self.n_patches, window, self.num_summary_tokens, True).unsqueeze(0).to(x.dtype).to(x.device)
                    self.addition_cache[self.n_patches] = addition

                attn = attn + addition

            # If requested, create or merge a causal mask (upper-triangular) so tokens cannot attend to future tokens
            if is_causal:
                causal = torch.triu(torch.ones((N, N), device=x.device, dtype=torch.bool), diagonal=1)
                causal_mask = torch.zeros((N, N), device=x.device, dtype=attn.dtype if hasattr(attn, 'dtype') else x.dtype)
                causal_mask = causal_mask.masked_fill(causal, float(-100.0))
                if attn_mask is None:
                    attn_mask = causal_mask
                else:
                    try:
                        attn_mask = attn_mask + causal_mask
                    except Exception:
                        attn_mask = attn_mask

            # apply attention mask if provided (handles windowed masks shaped like in WindowAttention)
            if attn_mask is not None:
                try:
                    nW = attn_mask.shape[0]
                    attn = attn.view(B // nW, nW, self.num_heads, N, N) + attn_mask.unsqueeze(1).unsqueeze(0)
                    attn = attn.view(-1, self.num_heads, N, N)
                except Exception:
                    attn = attn + attn_mask

            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
