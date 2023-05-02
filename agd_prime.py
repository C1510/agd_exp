import math
import torch

from torch.nn.init import orthogonal_, zeros_

class AGD:
    @torch.no_grad()
    def __init__(self, net, args, gain=1.0, wmult=1.0):

        self.net = net
        self.depth = 0
        self.gain = gain
        self.args = args

        groups = []
        curr = []
        for name, p in net.named_parameters():
            print(name)
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
            p *= self.singular_value(name, p) * wmult

    def get_block_scale(self, name):
        if 'resnet' not in self.args.arch:
            return 1
        else:
            if 'layer' not in name:
                return 1
            if 'shortcut' in name:
                return 1
            else:
                block_num = int(name.split('.')[0].split('layer')[-1])-1
                return len(self.net.num_blocks) ** (1/self.net.num_blocks[block_num])

    def singular_value(self, name, p):
        sv = math.sqrt(p.shape[0] / p.shape[1])
        if p.dim() == 4:
            sv /= math.sqrt(p.shape[2] * p.shape[3])
        sv /= math.sqrt(self.get_block_scale(name))
        return sv

    @torch.no_grad()
    def step(self):

        G = 0
        for name, p in self.net.named_parameters():
            if p.dim() != 1:
                G += self.singular_value(name, p) * p.grad.norm(dim=(0,1)).sum()
        G /= self.depth

        log = math.log(0.5 * (1 + math.sqrt(1 + 4*G)))

        for name in self.groups:
            p_main = self.params_dict[name[0]]
            update = log / self.depth * self.singular_value(name[0], p_main)
            p = self.params_dict[name[0]]
            p -= update * p.grad / p_main.grad.norm(dim=(0, 1), keepdim=True)
            denom = p_main.grad.norm()
            for name_ in name[1:]:
                p = self.params_dict[name_]
                p -= update * p.grad / denom

        return log
