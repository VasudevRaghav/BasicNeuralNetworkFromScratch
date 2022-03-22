import numpy as np

class NeuronLayerDense:
    def __init__(self,inputSize,neuronCount):
        self.weights = 0.1*np.random.randn(inputSize,neuronCount)
        self.baises = np.zeros((1,neuronCount))
    def forward(self,data):
        self.output = np.dot(data,self.weights)+self.baises

X = [[1.0,2.0,3.0,2.5],
     [2.0,5.0,-1.0,2.0],
     [-1.5,2.7,3.3,-0.8]]

layer1=NeuronLayerDense(len(X[0]),4)
layer1.forward(X)

layer2=NeuronLayerDense(len(layer1.output[0]),2)
layer2.forward(layer1.output)

print(layer1.output)
print(layer2.output)