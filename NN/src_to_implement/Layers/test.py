import numpy as np
weights = np.ones((6))
input_tensor = np.ones((3,4))
 
# for i in range(input_tensor.shape[0]):
#     input_tensor[i,:] = input_tensor[i,:] * weights
exp_tensor = np.exp(weights)
den = np.sum(exp_tensor, axis = 0)
den = np.repeat(np.expand_dims(den,axis=1),exp_tensor.shape[0],axis=1)
output_tensor = np.divide(exp_tensor, den)
