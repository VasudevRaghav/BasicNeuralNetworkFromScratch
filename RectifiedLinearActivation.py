import numpy as np

class NeuronLayerDense:
    def __init__(self,inputSize,neuronCount):
        self.output=[]
        self.weights = 0.1*np.random.randn(inputSize,neuronCount)
        self.baises = np.zeros((1,neuronCount))
    def forward(self,data):
        self.output=np.dot(data,self.weights)+self.baises

class Activation_Relu:
    def __init__(self):
        pass
    def __init__(self,data):
        self.forward(data)
    def forward(self,data):
        self.output=np.maximum(0,data)

X = [[1.0,2.0,3.0,2.5],
     [2.0,5.0,-1.0,2.0],
     [-1.5,2.7,3.3,-0.8]]

layer1=NeuronLayerDense(len(X[0]),4)
layer1.forward(X)

layer1=Activation_Relu(layer1.output)

print(layer1.output)