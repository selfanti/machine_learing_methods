import math

def compute_H_W_out(H_in,padding,dilation,kernel_size,stride):
    Hout=math.floor((H_in+2*padding-dilation*(kernel_size-1)-1)/stride+1)
    return Hout