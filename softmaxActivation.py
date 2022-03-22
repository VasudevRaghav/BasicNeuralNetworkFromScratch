import numpy as np
import nnfs
from nnfs.datasets import spiral_data
import pprint as pp

nnfs.init()

class NeuronLayerDense:
    def __init__(self,inputSize,neuronCount):
        self.output=[]
        self.weights = 0.1*np.random.randn(inputSize,neuronCount)
        self.baises = np.zeros((1,neuronCount))
    def forward(self,data):
        self.output=np.dot(data,self.weights)+self.baises

class Activation_ReLU:
    def forward(self,data):
        self.output=np.maximum(0,data)

class Activation_Softmax:
    def forward(self,data):
        exp_values = np.exp(data-np.max(data,axis=1,keepdims=True))
        self.output = exp_values/np.sum(exp_values,axis=1,keepdims=True)

X,Y = spiral_data(100,3)

layer1=NeuronLayerDense(2,8)
activation1=Activation_ReLU()

layer2=NeuronLayerDense(8,4)
activation2=Activation_Softmax()

layer1.forward(X)
activation1.forward(layer1.output)

layer2.forward(activation1.output)
activation2.forward(layer2.output)

pp.pprint(activation2.output)