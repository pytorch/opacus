class RDPAccountant:
    def __init__(self):
        self.steps = []

    def step(self, noise_multiplier, sample_rate):
        self.steps.append((noise_multiplier, sample_rate))

    def get_privacy_spent(self, delta, alphas):
        # TODO: well you know
        return 0, 1
