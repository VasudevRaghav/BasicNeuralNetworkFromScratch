import numpy as np
import nnfs
from nnfs.datasets import spiral_data

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

class Loss:
    def calculate(self,output,y):
        sample_loss = self.forward(output,y)
        data_loss = np.mean(sample_loss)
        return data_loss

class Loss_CategoricalCrossEntropy(Loss):
    def forward(self,y_pred,y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred,1e-7,1-1e-7)

        if len(y_true.shape)==1:
            correct_confidences = y_pred_clipped[range(samples),y_true]
        elif len(y_true.shape)==2:
            correct_confidences = np.sum(y_pred_clipped*y_true,axis=1)

        negative_log_likelihood = -np.log(correct_confidences)
        return negative_log_likelihood

X,Y = spiral_data(100,3)

layer1=NeuronLayerDense(2,3)
activation1=Activation_ReLU()

layer2=NeuronLayerDense(3,3)
activation2=Activation_Softmax()

layer1.forward(X)
activation1.forward(layer1.output)

layer2.forward(activation1.output)
activation2.forward(layer2.output)

lossOutput = Loss_CategoricalCrossEntropy()
loss=lossOutput.calculate(activation2.output,Y)

print("Loss:\n",loss)