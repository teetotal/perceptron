from perceptron import perceptron
import random

'''
index -1
- -> 1
+ -> -1
'''

dim = 7
def get_data(size):
    x_arr = []
    y_arr = []

    for i in range(size):
        x = [random.random() for _ in range(dim)]
        y = -1
        
        if i % 2 == 0:
            x[-1] *= -1
            y = 1 

        x_arr.append(x)
        y_arr.append(y)
    
    return x_arr, y_arr

p = perceptron(dim)
inputs, outpus = get_data(10000)
p.fit(inputs, outpus, 50)

inputs_eval, outputs_eval = get_data(100)
p.eval(inputs_eval, outputs_eval)
