import torch
__version = 1.0

def zeropower_via_newtonschulz5(G, steps: int):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert G.ndim >= 2 # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X
    
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def muon_update(grad, momentum, beta=0.95, ns_steps=5, nesterov=True):
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    if update.ndim == 4: # for the case of conv filters
        update = update.view(len(update), -1)
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    update *= max(1, grad.size(-2) / grad.size(-1))**0.5
    return update


def muon_update_wo_momentum_update(momentum, ns_steps=5):
    update = momentum.clone()
    if update.ndim == 4:
        update = update.view(len(update), -1)
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    update *= max(1, momentum.size(-2) / momentum.size(-1))**0.5
    return update


def adam_update(grad, buf1, buf2, step, betas, eps):
    # buf1.lerp_(grad, 1 - betas[0])
    # buf2.lerp_(grad.square(), 1 - betas[1])
    # buf1c = buf1 / (1 - betas[0]**step)
    # buf2c = buf2 / (1 - betas[1]**step)
    # return buf1c / (buf2c.sqrt() + eps)

    return grad


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-2, momentum=0.0,weight_decay=0.0, nesterov=False):
        if lr < 0.0:
            raise ValueError(f"Invalid lr: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if nesterov and (momentum <= 0.0):
            raise ValueError("Nesterov requires momentum>0")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]
            nesterov = group["nesterov"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                d_p = p.grad

                if momentum != 0.0:
                    state = self.state[p]
                    buf = state.get("momentum_buffer", None)
                    if buf is None:
                        buf = torch.clone(d_p).detach()
                        state["momentum_buffer"] = buf
                    else:
                        buf.mul_(momentum).add_(d_p)

                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf

                if weight_decay != 0.0:
                    p.mul_(1 - lr * weight_decay)
                p.add_(d_p, alpha=-lr)

        return loss

class SF_SGD1(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-2, momentum=0.0,weight_decay=0.0, nesterov=False, beta=0.9):
        if lr < 0.0:
            raise ValueError(f"Invalid lr: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if nesterov and (momentum <= 0.0):
            raise ValueError("Nesterov requires momentum>0")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            weight_decay=weight_decay,
            beta=beta,
            t=1,
            c=0.5
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]
            nesterov = group["nesterov"]
            beta = group['beta']
            t = group['t']
            c = group['c']

            for p in group["params"]:
                if p.grad is None:
                    continue
                d_p = p.grad

                if momentum != 0.0:
                    state = self.state[p]
                    buf = state.get("momentum_buffer", None)
                    x = state.get("x", None)
                    y = state.get("y", None)
                    z = state.get("z", None)
                    if buf is None:
                        buf = torch.clone(d_p).detach()
                        x = torch.clone(p).detach()
                        y = torch.clone(p).detach()
                        z = torch.clone(p).detach()
                        state["momentum_buffer"] = buf
                        state['x'], state['y'], state['z'] = x, y ,z
                    else:
                        buf.mul_(momentum).add_(d_p)

                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf

                state['z'].add_(d_p, alpha=-lr)
                state['x'] = (1 - c) * state['x'] + c * state['z']
                state['y'] = (1 - beta) * state['z'] + beta * state['x']

                p.copy_(state['y'])
            group['t'] += 1
            group['c'] = 1 / (group['t'] + 1)

        return loss
    
    def exchange(self, point):
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if point in state.keys():
                    p.copy_(state[point])


class SF_SGD2(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-2, momentum=0.0,weight_decay=0.0, nesterov=False, beta=0.9):
        if lr < 0.0:
            raise ValueError(f"Invalid lr: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if nesterov and (momentum <= 0.0):
            raise ValueError("Nesterov requires momentum>0")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            weight_decay=weight_decay,
            beta=beta,
            t=1,
            c=0.5
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]
            nesterov = group["nesterov"]
            beta = group['beta']
            t = group['t']
            c = group['c']

            for p in group["params"]:
                if p.grad is None:
                    continue
                d_p = p.grad

                if momentum != 0.0:
                    state = self.state[p]
                    buf = state.get("momentum_buffer", None)
                    m = state.get('m',None)
                    if buf is None:
                        buf = torch.clone(d_p).detach()
                        m = torch.clone(d_p).detach()
                        state["momentum_buffer"] = buf
                        state["m"] = m
                    else:
                        buf.mul_(momentum).add_(d_p)
                        m.mul_(1 - 1 / t).add_(buf)

                    if nesterov:
                        d_p = d_p.add(beta / (t+1) * m + (1 - beta) * buf, alpha=momentum)
                    else:
                        d_p = beta / (t+1) * m + (1 - beta) * buf

                p.add_(d_p, alpha=-lr)


            group['t'] += 1
            group['c'] = 1 / (group['t'] + 1)

        return loss

class DM_SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-2, momentum=0.0,weight_decay=0.0, nesterov=False, beta=0.9):
        if lr < 0.0:
            raise ValueError(f"Invalid lr: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if nesterov and (momentum <= 0.0):
            raise ValueError("Nesterov requires momentum>0")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            weight_decay=weight_decay,
            beta=beta,
            t=1,
            c=0.5
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]
            nesterov = group["nesterov"]
            beta = group['beta']
            t = group['t']
            c = group['c']

            for p in group["params"]:
                if p.grad is None:
                    continue
                d_p = p.grad

                if momentum != 0.0:
                    state = self.state[p]
                    m = state.get('m', None)
                    n = state.get('n', None)
                    if m is None:
                        m = torch.clone(d_p).detach()
                        n = torch.clone(d_p).detach()
                        state["m"] = m
                        state['n'] = n
                    else:
                        m.add_(n).mul_((t-1)/t).add_(d_p)
                        n.mul_((t-1) / t).add_(d_p).mul_(momentum)

                d_p = beta / (t+1) * m + (1 - beta) * n

                if weight_decay != 0.0:
                    p.mul_(1 - lr * weight_decay)

                p.add_(d_p, alpha=-lr)


            group['t'] += 1
            group['c'] = 1 / (group['t'] + 1)

        return loss


import mpmath as mp

def func(mu, t):
    mu_m = mp.mpf(mu)
    t_m  = mp.mpf(t)

    if t_m < 1:
        raise ValueError("t must be >= 1")
    if mu_m == 1:
        raise ValueError("mu must be != 1")

    a = mp.log(mu_m)
    z = mp.e * mu_m**(-(t_m + 1))
    w = mp.lambertw(z, 0)
    i_star = mp.re((1 - w) / a)

    i_int = int(mp.floor(i_star))
    t_int = int(mp.floor(t_m))
    i_int = max(1, min(t_int, i_int))

    # compute f at integer i, return as Python float
    f_val = (mp.mpf(i_int)/t_m) * (1 - mu_m**(t_m + 1 - mp.mpf(i_int))) / (1 - mu_m)
    return i_int, float(mp.re(f_val))

class DM_SGD2(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-2, momentum=0.0,weight_decay=0.0, nesterov=False, beta=0.9):
        if lr < 0.0:
            raise ValueError(f"Invalid lr: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if nesterov and (momentum <= 0.0):
            raise ValueError("Nesterov requires momentum>0")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            weight_decay=weight_decay,
            beta=beta,
            t=1,
            c=0.5
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]
            nesterov = group["nesterov"]
            beta = group['beta']
            t = group['t']
            c = group['c']

            for p in group["params"]:
                if p.grad is None:
                    continue
                d_p = p.grad

                if momentum != 0.0:
                    state = self.state[p]
                    m = state.get('m', None)
                    n = state.get('n', None)
                    if m is None:
                        m = torch.clone(d_p).detach()
                        n = torch.clone(d_p).detach()
                        state["m"] = m
                        state['n'] = n
                    else:
                        m.mul_((t-1)/t).add_(d_p)
                        n.mul_((t-1) / t).add_(d_p).mul_(momentum)
                i_star, f_star = func(momentum, t)
                d_p = beta * t / (t+1) * f_star / i_star * m + (1 - beta) * n
                p.add_(d_p, alpha=-lr)


            group['t'] += 1
            group['c'] = 1 / (group['t'] + 1)

        return loss


class DM_SGD3(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-2, momentum=0.0,weight_decay=0.0, nesterov=False, beta=0.9):
        if lr < 0.0:
            raise ValueError(f"Invalid lr: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if nesterov and (momentum <= 0.0):
            raise ValueError("Nesterov requires momentum>0")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            weight_decay=weight_decay,
            beta=beta,
            t=1,
            c=0.5
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]
            nesterov = group["nesterov"]
            beta = group['beta']
            t = group['t']
            c = group['c']

            for p in group["params"]:
                if p.grad is None:
                    continue
                d_p = p.grad

                if momentum != 0.0:
                    state = self.state[p]
                    m = state.get('m', None)
                    n = state.get('n', None)
                    if m is None:
                        m = torch.clone(d_p).detach()
                        n = torch.clone(d_p).detach()
                        state["m"] = m
                        state['n'] = n
                    else:
                        m.mul_((t-1)/t).add_(d_p)
                        n.mul_((t-1) / t).add_(d_p).mul_(momentum)
                d_p = beta  / (t+1)  * (1 - momentum ** t) / (1 - momentum) * m + (1 - beta) * n
                p.add_(d_p, alpha=-lr)


            group['t'] += 1
            group['c'] = 1 / (group['t'] + 1)

        return loss
    

class SingleDeviceMuonWithAuxAdam(torch.optim.Optimizer):
    """
    Non-distributed variant of MuonWithAuxAdam.
    """
    def __init__(self, param_groups):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                # defaults
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(["params", "lr", "momentum", "weight_decay", "use_muon"])
            else:
                # defaults
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(["params", "lr", "betas", "eps", "weight_decay", "use_muon"])
        super().__init__(param_groups, dict())

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                for p in group["params"]:
                    if p.grad is None:
                        # continue
                        p.grad = torch.zeros_like(p)  # Force synchronization
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"],nesterov=False)
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update.reshape(p.shape), alpha=-group["lr"])
            else:
                for p in group["params"]:
                    if p.grad is None:
                        # continue
                        p.grad = torch.zeros_like(p)  # Force synchronization
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(p.grad, state["exp_avg"], state["exp_avg_sq"],
                                         state["step"], group["betas"], group["eps"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        return loss
    


class DM_Muon(torch.optim.Optimizer):
    """
    Non-distributed variant of MuonWithAuxAdam.
    """
    def __init__(self, param_groups, beta=0.9):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                # defaults
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                group['beta'] = beta
            else:
                # defaults
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)
                group["momentum"] = group.get("momentum", 0.95)
                group['beta'] = beta
        super().__init__(param_groups, dict())

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                for p in group["params"]:
                    if p.grad is None:
                        # continue
                        p.grad = torch.zeros_like(p)  # Force synchronization
                    state = self.state[p]
                    if len(state) == 0:
                        state["m"] = torch.zeros_like(p)
                        state["n"] = torch.zeros_like(p)
                        state['t'] = 0
                    state["t"] += 1
                    t = state["t"]
                    m = state["m"]
                    n = state["n"]
                    beta = group['beta']
                    momentum = group["momentum"]

                    m.mul_((t-1)/t).add_(p.grad)
                    n.mul_((t-1) / t).add_(p.grad).mul_(momentum)
                    momentum_buffer = (beta  / (t+1)  * (1 - momentum ** t) / (1 - momentum) * m + (1 - beta) * n)
                    update = muon_update_wo_momentum_update(momentum_buffer)
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update.reshape(p.shape), alpha=-group["lr"])
            else:
                # for p in group["params"]:
                #     if p.grad is None:
                #         # continue
                #         p.grad = torch.zeros_like(p)  # Force synchronization
                #     state = self.state[p]
                #     if len(state) == 0:
                #         state["exp_avg"] = torch.zeros_like(p)
                #         state["exp_avg_sq"] = torch.zeros_like(p)
                #         state["step"] = 0
                #     state["step"] += 1
                #     update = adam_update(p.grad, state["exp_avg"], state["exp_avg_sq"],
                #                          state["step"], group["betas"], group["eps"])
                #     p.mul_(1 - group["lr"] * group["weight_decay"])
                #     p.add_(update, alpha=-group["lr"])
                for p in group["params"]:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)  # Force synchronization
                    state = self.state[p]
                    if len(state) == 0:
                        state["m"] = torch.zeros_like(p)
                        state["n"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1  
                    t = state["step"]
                    m = state["m"]
                    n = state["n"]
                    beta = group['beta']
                    momentum = group["momentum"]
                    m.mul_((t-1)/t).add_(p.grad)
                    n.mul_((t-1) / t).add_(p.grad).mul_(momentum)
                    momentum_buffer = (beta  / (t+1)  * (1 - momentum ** t) / (1 - momentum) * m + (1 - beta) * n)
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(momentum_buffer, alpha=-group["lr"])

        return loss