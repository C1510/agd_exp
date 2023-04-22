import math
import torch

from torch.nn.init import orthogonal_, zeros_

def get_depth_val(name):
    if 'c_fc' in name:
        return 1
    elif 'attn' in name:
        return 3
    else:
        return 1

def singular_value(name, p):
    if get_depth_val(name) == 1:
        if ('transformer.wte.weight' in name) or ('transformer.wpe.weight' in name):
            sv = math.sqrt(p.shape[1] / p.shape[0])
        #if 'lm_head' in name:
        #    sv = math.sqrt(p.shape[0] / p.shape[1])
        #else:
        sv = math.sqrt(p.shape[0] / p.shape[1])
    else:
        sv = math.sqrt(p.shape[0] / (3*p.shape[1]) )
    if p.dim() == 4:
        sv /= math.sqrt(p.shape[2] * p.shape[3])
    return sv

class AGD:
    @torch.no_grad()
    def __init__(self, net, gain=1.0, wmult=1.0):

        self.net = net
        self.depth = 0
        self.gain = gain
        #self.gain = 0.1

        groups = []
        curr = []
        for name, p in net.named_parameters():

            if 'ln_' in name:
                continue
            if 'weight' in name and p.dim() == 2:
                groups.append(curr)
                curr = [name]
                self.depth += get_depth_val(name)
            elif 'weight' in name and p.dim() == 4:
                groups.append(curr)
                curr = [name]
                self.depth += get_depth_val(name)
            else:
                curr.append(name)
        if curr != groups[-1]:
            groups.append(curr)
        if groups[0] == []:
            groups = groups[1:]
        self.groups = groups
        self.params_dict = dict(net.named_parameters())

        for name, p in net.named_parameters():
            #print(name, p.shape)
            if 'ln_' in name:
                continue
            if 'lm_' in name:
                continue
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
            p *= singular_value(name, p) * wmult
            print(name, p.shape, singular_value(name, p))

        print('DEPTH: ',self.depth)
        print(self.groups)
        # self.depth *= 2

    @torch.no_grad()
    def step(self):

        G = 0
        for name, p in self.net.named_parameters():
            if 'ln_' in name:
                continue
            if p.dim() != 1:
                #print(name, p.shape, singular_value(name, p), p.grad.norm(dim=(0,1)).sum(),  singular_value(name, p) * p.grad.norm(dim=(0,1)).sum())
                G += singular_value(name, p) * p.grad.norm(dim=(0,1)).sum()
        G /= self.depth

        log = self.gain*math.log(0.5 * (1 + math.sqrt(1 + 4*G)))

        for name in self.groups:
            p_main = self.params_dict[name[0]]
            update = log / self.depth * singular_value(name, p_main)
            p = self.params_dict[name[0]]
            p -= update * p.grad / p_main.grad.norm(dim=(0, 1), keepdim=True)
            denom = p_main.grad.norm()
            for name_ in name[1:]:
                p = self.params_dict[name_]
                p -= update * p.grad / denom
        #print('LOG', log)
        #sys.exit()
        return log
