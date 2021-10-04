import chainer

state_keys = ["loss_scale"]


class IdentityLossScalingHook(chainer.function_node.FunctionNode):
    """ Its backward function will attach the loss scale """

    def __init__(self, state, pre_hook=True):
        super().__init__()
        self.state = state  # a dict
        self.pre_hook = pre_hook

    def forward(self, inputs):
        return inputs

    def backward(self, indexes, grad_outputs):
        """ Behavior depends on the setting """
        gs = []

        # HACK: fix for concat
        if self.pre_hook and len(self.state) == 1:
            for i in range(1, len(grad_outputs)):
                self.state[i] = self.state[0]

        for i, g in enumerate(grad_outputs):
            if g is not None:
                if self.pre_hook:  # attach state
                    for k, v in self.state[i].items():
                        g.__dict__[k] = v
                else:  # store state
                    if i not in self.state:
                        self.state[i] = {}
                    for k in state_keys:
                        self.state[i][k] = g.__dict__[k]
            gs.append(g)

        return gs


class IdentityLossScalingWrapper(chainer.link.Link):
    """ Wraps a function and passes the loss scaling. """

    def __init__(self, func):
        """ func is what to be wrapped. """
        super().__init__()

        self.func = func
        self.state = {}

    def forward(self, *args):
        """ Passes identical loss scale """
        # HACK: fix the case that args[0] is a list (concat)
        if isinstance(args[0], list):
            ys = IdentityLossScalingHook(self.state).apply(tuple(args[0]))
            zs = self.func(ys)
        else:
            ys = IdentityLossScalingHook(self.state).apply(args)
            zs = self.func(*ys)  # execute the function

        if not isinstance(zs, tuple):
            zs = (zs,)

        rs = IdentityLossScalingHook(self.state, pre_hook=False).apply(zs)
        if len(rs) == 1:
            rs = rs[0]
        return rs
