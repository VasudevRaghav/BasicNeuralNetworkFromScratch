import numpy as np

data = [[1.0,2.0,3.0,2.5],
        [2.0,5.0,-1.0,2.0],
        [-1.5,2.7,3.3,-0.8]]

weights1 = [[0.2,0.8,-0.5,1.0],
           [0.5,-0.91,0.26,-0.5],
           [-0.26,-0.27,0.17,0.87]]

baises1 = [2.0,3.0,0.5]

weights2 = [[0.1,-0.14,0.5],
           [-0.5,0.12,-0.33],
           [-0.44,0.73,-0.13]]

baises2 = [-1,2,-0.5]

layer1_output = np.dot(data,np.array(weights1).T)+baises1
layer2_output = np.dot(layer1_output,np.array(weights2).T)+baises2

print(layer2_output)