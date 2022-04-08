# perceptron

perceptron implementation of pure python

## formula
If ∑ wixi >= 0 : 1

else: -1


∑ wixi = bias + w1x1 + .... + wixi

bias = w0x0

x0 = 1

## update
wi := wi + (η * (y-prediction) * xi)