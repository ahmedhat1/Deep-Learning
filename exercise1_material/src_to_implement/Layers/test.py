import numpy as np
weights = np.ones((3,3))
input_tensor = np.ones((3,3,3))  * 2
 
# for i in range(input_tensor.shape[0]):
#     input_tensor[i,:] = input_tensor[i,:] * weights
print("weights " , weights)
print("input tensor " , input_tensor)
print(np.dot(input_tensor,weights))