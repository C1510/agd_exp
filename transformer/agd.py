import math, sys
import torch

from torch.nn.init import orthogonal_, zeros_
#config.n_layer
def get_depth_val(name, n_layer):
    if '_attn' in name:
        return 3
    elif 'c_proj.weight' in name:
        return n_layer
    else:
        return 1


def singular_value(name, p, n_layer):
    depth = get_depth_val(name, n_layer)
    #if ('transformer.wte.weight' in name) or ('transformer.wpe.weight' in name):
        #sv = math.sqrt(p.shape[1] / (depth*p.shape[0]))
    #else:
    sv = math.sqrt(p.shape[0] / (depth*p.shape[1]))
    return sv


class AGD:
    @torch.no_grad()
    def __init__(self, net, gain=1.0, wmult=1.0):

        self.net = net
        self.depth = 0
        self.gain = gain
        #self.gain = 0.2
        self.n_layer = self.net.config.n_layer
        groups = []
        curr = []
        for name, p in net.named_parameters():
            #print(name)
            if 'ln_' in name:
                continue
            if 'weight' in name and p.dim() == 2:
                groups.append(curr)
                curr = [name]
                self.depth += get_depth_val(name, self.n_layer)
            else:
                curr.append(name)
        if curr != groups[-1]:
            groups.append(curr)
        if groups[0] == []:
            groups = groups[1:]
        self.groups = groups
        self.params_dict = dict(net.named_parameters())
        #sys.exit()

        for name, p in net.named_parameters():
            if 'ln_' in name:
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
            print(name, singular_value(name, p, self.n_layer))
            p *= singular_value(name, p, self.n_layer) * wmult

        print('DEPTH: ',self.depth)
        print(self.groups)

    @torch.no_grad()
    def step(self):

        G = 0
        for name, p in self.net.named_parameters():
            if 'ln_' in name:
                continue
            if p.dim() != 1:
                #print(name, p.shape, singular_value(name, p), p.grad.norm(dim=(0,1)).sum(),  singular_value(name, p) * p.grad.norm(dim=(0,1)).sum())
                try:
                    G += singular_value(name, p, self.n_layer) * p.grad.norm(dim=(0,1)).sum()
                except:
                    pass
        G /= self.depth

        log = self.gain*math.log(0.5 * (1 + math.sqrt(1 + 4*G)))

        for name in self.groups:
            p_main = self.params_dict[name[0]]
            update = log / self.depth * singular_value(name, p_main, self.n_layer)
            p = self.params_dict[name[0]]
            try:
                p -= update * p.grad / p_main.grad.norm(dim=(0, 1), keepdim=True)
            except:
                continue
            denom = p_main.grad.norm()
            for name_ in name[1:]:
                p = self.params_dict[name_]
                p -= update * p.grad / denom

        return log
