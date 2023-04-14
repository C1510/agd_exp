import math
import torch

from torch.nn.init import orthogonal_, zeros_

def singular_value(p):
    sv = math.sqrt(p.shape[0] / p.shape[1])
    if p.dim() == 4:
        sv /= math.sqrt(p.shape[2] * p.shape[3])
    return sv

class AGD:
    @torch.no_grad()
    def __init__(self, net, gain=1.0, wmult=1.0):

        self.net = net
        self.depth = 0
        self.gain = gain

        groups = []
        curr = []
        for name, p in net.named_parameters():
            if 'weight' in name and p.dim() == 2:
                groups.append(curr)
                curr = [name]
                self.depth += 1
            elif 'weight' in name and p.dim() == 4:
                groups.append(curr)
                curr = [name]
                self.depth += 1
            else:
                curr.append(name)
        if curr != groups[-1]:
            groups.append(curr)
        if groups[0] == []:
            groups = groups[1:]
        self.groups = groups
        self.params_dict = dict(net.named_parameters())

        for name, p in net.named_parameters():
            if p.dim() == 1 and 'weight' in name:
                continue
            if p.dim() == 1 and 'bias' in name:
                zeros_(p)
                continue
            if p.dim() == 2:
                orthogonal_(p)
            if p.dim() == 4:
                for kx in range(p.shape[2]):
                    for ky in range(p.shape[3]):
                        orthogonal_(p[:, :, kx, ky])
            p *= singular_value(p) * wmult

    @torch.no_grad()
    def step(self):

        G = 0
        for p in self.net.parameters():
            if p.dim() != 1:
                G += singular_value(p) * p.grad.norm(dim=(0,1)).sum()
        G /= self.depth

        log = math.log(0.5 * (1 + math.sqrt(1 + 4*G)))

        for name in self.groups:
            p_main = self.params_dict[name[0]]
            update = log / self.depth * singular_value(p_main)
            p = self.params_dict[name[0]]
            p -= update * p.grad / p_main.grad.norm(dim=(0, 1), keepdim=True)
            denom = p_main.grad.norm()
            for name_ in name[1:]:
                p = self.params_dict[name_]
                p -= update * p.grad / denom

        return log
