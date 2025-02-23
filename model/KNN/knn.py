import  numpy as np
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
import kagglehub

# Download latest version
path = kagglehub.dataset_download("wenruliu/adult-income-dataset")

print("Path to dataset files:", path)
def distance_fc(a,b,p):
    if len(a) !=len(b):
        raise  ValueError('length of a and b must be equal')
    else:
        return np.sum(np.abs(a-b)**p)**(1/p)

print()