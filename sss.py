import torch
import torch.nn as nn
from einops import rearrange


class SelfScaleShift(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ss = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim * 2)
        )

    def forward(self, x):
        x = rearrange(x, 'b c ... -> b ... c')
        scale, shift = self.ss(x).chunk(2, dim=-1)
        x = (scale + 1) * x + shift
        x = rearrange(x, 'b ... c -> b c ...')
        return x


if __name__ == '__main__':
    a = torch.rand(2, 3, 4, 4)
    sss = SelfScaleShift(3)
    print(f'a: {a}')
    print(f'out: {sss(a)}')
