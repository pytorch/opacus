from .optimizer import DPOptimizer


class _NoiseScheduler(object):
    def __init__(self, optimizer):
        # Attach optimizer
        if not isinstance(optimizer, DPOptimizer):
            raise TypeError("{} is not a DPOptimizer".format(type(optimizer).__name__))
        self.optimizer = optimizer

        self.step()

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.
        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {
            key: value for key, value in self.__dict__.items() if key != "optimizer"
        }

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.
        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_noise_multiplier(self):
        # Compute learning rate using chainable form of the scheduler
        raise NotImplementedError

    def step(self):
        noise_multiplier = self.get_noise_multiplier()
        self.optimizer.privacy_engine.noise_mutliplier = noise_multiplier
