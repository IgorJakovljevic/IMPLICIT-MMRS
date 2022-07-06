
class EvalResults():

    def __init__(self, name, confidence, metrics, factors, alpha, iterations):
        self.name = name
        self.confidence = confidence
        self.metrics = metrics
        self.factors = factors
        self.alpha = alpha
        self.iterations = iterations