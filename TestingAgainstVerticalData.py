import nnfs
import matplotlib.pyplot as graph
import numpy as np
from nnfs.datasets import vertical_data
nnfs.init()
X,Y=vertical_data(400,3)
graph.scatter(X[:,0],X[:,1],c=Y,s=4, cmap='brg')
graph.plot()

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

Lowest_Loss=loss
Best_Weight_Layer1=layer1.weights.copy()
Best_Bais_Layer1=layer1.baises.copy()
Best_Weight_Layer2=layer2.weights.copy()
Best_Bais_layer2=layer2.baises.copy()

for i in range(100000):
    newWeight1=0.05*np.random.randn(2,3)
    layer1.weights+=newWeight1
    newBais1=0.05*np.random.randn(1,3)
    layer1.baises+=newBais1
    newWeight2=0.05*np.random.randn(3,3)
    layer2.weights+=newWeight2
    newBais2=0.05*np.random.randn(1,3)
    layer2.baises+=newBais2
    
    layer1.forward(X)
    activation1.forward(layer1.output)
    layer2.forward(activation1.output)
    activation2.forward(layer2.output)    
    loss=lossOutput.calculate(activation2.output,Y)
    
    if loss<Lowest_Loss:
        Best_Weight_Layer1=layer1.weights.copy()
        Best_Bais_Layer1=layer1.baises.copy()
        Best_Weight_Layer2=layer2.weights.copy()
        Best_Bais_layer2=layer2.baises.copy()
        Lowest_Loss=loss
        print("New weights and baises found at iteration:",i,"Loss:",Lowest_Loss)
    else:
        layer1.weights=Best_Weight_Layer1.copy()
        layer1.baises=Best_Bais_Layer1.copy()
        layer2.weights=Best_Weight_Layer2.copy()
        layer2.baises=Best_Bais_layer2.copy()

def classPrediction(testX):
    dense1=np.dot(np.array(testX),Best_Weight_Layer1)+Best_Bais_Layer1
    activation1=np.maximum(0,dense1)
    dense2=np.dot(activation1,Best_Weight_Layer2)+Best_Bais_layer2
    activation2_Exp_Value=np.exp(dense2-np.max(dense2,axis=1,keepdims=True))
    activationOutput=activation2_Exp_Value/np.sum(activation2_Exp_Value,axis=1,keepdims=True)
    prediction=np.argmax(activationOutput)
    return prediction+1

#testing trained neurons
testX1=[0.0,0.1]
print(classPrediction(testX1))
testX2=[0.3,0.7]
print(classPrediction(testX2))
testX3=[0.7,0.3]
print(classPrediction(testX3))
#and more testing
print(classPrediction([0.8,0.3]))
print(classPrediction([0.5024236382,0.5]))
print(classPrediction([0.69,0.42]))