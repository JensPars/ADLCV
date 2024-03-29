import numpy as np
import cv2
import scipy.sparse
from scipy.sparse.linalg import spsolve
from copy import deepcopy
from os import path
import random

def laplacian_matrix(n, m):
    """Generate the Poisson matrix. 
    Refer to: 
    https://en.wikipedia.org/wiki/Discrete_Poisson_equation
    Note: it's the transpose of the wiki's matrix 
    """
    mat_D = scipy.sparse.lil_matrix((m, m))
    mat_D.setdiag(-1, -1)
    mat_D.setdiag(4)
    mat_D.setdiag(-1, 1)
        
    mat_A = scipy.sparse.block_diag([mat_D] * n).tolil()
    
    mat_A.setdiag(-1, 1*m)
    mat_A.setdiag(-1, -1*m)
    
    return mat_A


def poisson_edit(source, target, mask):
    target=deepcopy(target)
    y_max, x_max = target.shape[:-1]
    y_min, x_min = 0, 0

    x_range = x_max - x_min
    y_range = y_max - y_min
    mask = mask[y_min:y_max, x_min:x_max]    
    mask[mask != 0] = 1
    mat_A = laplacian_matrix(y_range, x_range)
    laplacian = mat_A.tocsc()
    for y in range(1, y_range - 1):
        for x in range(1, x_range - 1):
            if mask[y, x] == 0:
                k = x + y * x_range
                mat_A[k, k] = 1
                mat_A[k, k + 1] = 0
                mat_A[k, k - 1] = 0
                mat_A[k, k + x_range] = 0
                mat_A[k, k - x_range] = 0
    mat_A = mat_A.tocsc()

    mask_flat = mask.flatten()    
    for channel in range(source.shape[2]):
        source_flat = source[y_min:y_max, x_min:x_max, channel].flatten()
        target_flat = target[y_min:y_max, x_min:x_max, channel].flatten()        
        alpha = 1
        mat_b = laplacian.dot(source_flat)*alpha
        mat_b[mask_flat==0] = target_flat[mask_flat==0]
        
        x = spsolve(mat_A, mat_b)
        x = x.reshape((y_range, x_range))
        x[x > 255] = 255
        x[x < 0] = 0
        x = x.astype('uint8')
        target[y_min:y_max, x_min:x_max, channel] = x

    return target

def blend_image(dst_img,src_img,composed_mask,cp_method):
    cp_method=random.sample(cp_method,1)[0]
    if cp_method=='basic':
        src_img=src_img[:3]
        return dst_img*(1-composed_mask)+src_img*composed_mask
    if cp_method=='alpha':
        assert src_img.shape[0]==4
        alpha=src_img[3:]/255
        src_img=src_img[:3]
        return dst_img*(1-alpha)+src_img*alpha
    if cp_method=='gaussian':
        src_img=src_img[:3]
        composed_mask=cv2.blur(composed_mask.astype('float32'),(5,5))
        return dst_img*(1-composed_mask)+src_img*composed_mask
    if cp_method=='possion':
        src_img=src_img[:3].transpose(1,2,0)
        dst_img=dst_img.transpose(1,2,0)
        return poisson_edit(src_img,dst_img,composed_mask).transpose(2,0,1)