data = [1,2,3,2.5]

weights = [[0.2,0.8,-0.5,1.0],
            [0.5,-0.91,0.26,-0.5],
            [0.26,-0.27,0.17,0.87]]

baises = [2,3,0.5]

layerOutput=[]
for neuronWeights,neuronBais in zip(weights,baises):
    neuronOutput=0
    for val, weight in zip(data,neuronWeights):
        neuronOutput += val*weight
    neuronOutput+=neuronBais
    layerOutput.append(neuronOutput)

print(layerOutput)