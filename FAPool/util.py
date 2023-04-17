
from numpy import *

def fspecial(func_name,kernel_size=3,sigma=1):
    if func_name=='gaussian':
        m=n=(kernel_size-1.)/2.
        y,x=ogrid[-m:m+1,-n:n+1]
        h=exp( -(x*x + y*y) / (2.*sigma*sigma) )
        h[ h < finfo(h.dtype).eps*h.max() ] = 0
        sumh=h.sum()
        if sumh!=0:
            h/=sumh
        return h
    
# import torch    
# print(torch.tensor(fspecial('gaussian',kernel_size=2,sigma=0.5)).size())
