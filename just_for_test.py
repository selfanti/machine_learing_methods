import pandas as pd
import torch
test_tensor=torch.tensor([1,2,3])
sig_tensor=torch.sigmoid(test_tensor)
print(test_tensor)
print(sig_tensor)