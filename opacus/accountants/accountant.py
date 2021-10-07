

class Accountant:
    def __init__(self):
        pass

    def step(self, noise_multiplier: float, sample_rate: float):
        raise NotImplementedError()

    def get_privacy_spent(self, delta: float):
        raise NotImplementedError()
