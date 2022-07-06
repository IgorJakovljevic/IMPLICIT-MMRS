
import math

class MeasureOfConfidence():
    def calculate(self):
        pass

    def name(self):
        pass

    def preprocess_cui(self, cui, df):
        pass

class BinaryMeasureOfConfidence(MeasureOfConfidence):
    def calculate(self, r_ui, alpha):
        return 1

    def name(self):
        return "Binary Measure < 1 >"

class SimpleMeasureOfConfidence(MeasureOfConfidence):
    def calculate(self, r_ui, alpha):
        return 1 + alpha*r_ui

    def name(self):
        return "Simple Measure < 1 + alpha*r_ui >"

class LogMeasureOfConfidence(MeasureOfConfidence):
    def calculate(self, r_ui, alpha):
        return 1 + alpha*(math.log(1+(r_ui/0.0001)))

    def name(self):
        return "Log Measure < 1 + alpha*log(1 + r_ui/epsilon) >"
