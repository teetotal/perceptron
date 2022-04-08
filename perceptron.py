import random

class perceptron:
    def __init__(self, dim, eta=0.001) -> None:
        self.eta = eta
        self.w = [ random.random() for _ in range(dim + 1)]

    def prediction(self, x:list) -> int:
        z = 0
        for i in range(len(x)):
            z += (self.w[i] * x[i])
        
        return self.phi(z)

    def phi(self, z) -> int:
        if z >= 0: return 1
        else: return -1

    def update_w(self, x, y) -> float:
        prediction = self.prediction(x)
        u = self.eta * (y - prediction)

        for i in range(len(x)):
            self.w[i] += (u * x[i])
        
        return u
    
    def add_bias(self, x):
        ret = [1]
        ret.extend(x)

        return ret

    def fit(self, inputs: list, outputs:list, epochs: int):
        inputs = [self.add_bias(x) for x in inputs]
        
        for epoch in range(epochs):
            loss = 0
            for i in range(len(outputs)):
                x = inputs[i]
                y = outputs[i]

                loss += self.update_w(x, y)
            
            if loss == 0: return
            print("fit", "epoch", epoch, "loss", loss)
    
    def eval(self, inputs: list, outputs:list):
        inputs = [self.add_bias(x) for x in inputs]
        acc = 0
        for i in range(len(outputs)):
            x = inputs[i]
            y = outputs[i]

            if self.prediction(x) == y: acc += 1
        
        print("evaluation", "accuracy", acc / len(outputs))

