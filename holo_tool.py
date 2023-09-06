import math

import numpy as np

import numpy
from scipy.io import loadmat


def fuckDataZL(data,path):
    great = loadmat(path)["data"]
    print(np.allclose(data, great))
    print(f"the x0's max is : {np.max(data)}, min is : {np.min(data)} shape is :{data.shape}")
    return np.allclose(data, great)

def Propagator_function(N0, z, lmda, screen):
    u = np.zeros((N0, N0),np.complex)
    for ii in range(N0):
        for jj in range(N0):
            # a = round(lmda * (ii-N0 / 2-1) / screen,4)
            # b = round(lmda * (jj-N0 / 2-1) / screen,4)
            a = lmda * (ii+1-N0 / 2-1) / screen
            b = lmda * (jj+1-N0 / 2-1) / screen
            if (pow(a,2) + pow(b,2)) <= 1:
                u[ii, jj] = np.exp(-2 * np.pi * 1j * z * np.sqrt(1 - pow(a,2) - pow(b,2)) / lmda )

    return u


def meshgrid(x):
    y = x
    XX, YY = np.meshgrid(x, y)
    return (XX,YY)


import numpy as np
import math
import copy


def fft1(src, dst=None):
    '''
    src: list is better.One dimension.
    '''
    l = len(src)
    n = int(math.log(l, 2))

    bfsize = np.zeros((l), dtype="complex")

    for i in range(n + 1):
        if i == 0:
            for j in range(l):
                bfsize[j] = src[Dec2Bin_Inverse2Dec(j, n)]
        else:
            tmp = copy.copy(bfsize)
            for j in range(l):
                pos = j % (pow(2, i))
                if pos < pow(2, i - 1):
                    bfsize[j] = tmp[j] + tmp[j + pow(2, i - 1)] * np.exp(complex(0, -2 * np.pi * pos / pow(2, i)))
                    bfsize[j + pow(2, i - 1)] = tmp[j] - tmp[j + pow(2, i - 1)] * np.exp(
                        complex(0, -2 * np.pi * pos / (pow(2, i))))
    return bfsize


def ifft1(src):
    for i in range(len(src)):
        src[i] = complex(src[i].real, -src[i].imag)

    res = fft1(src)

    for i in range(len(res)):
        res[i] = complex(res[i].real, -res[i].imag)

    return res / len(res)


def Dec2Bin_Inverse2Dec(n, m):
    '''
    Especially for fft.To find position.
    '''
    b = bin(n)[2:]
    if len(b) != m:
        b = "0" * (m - len(b)) + b
    b = b[::-1]
    return int(b, 2)




def FT(inData):
    [Nx,Ny] = inData.shape
    h = np.zeros((Nx, Ny),np.complex)

    aaa = list(range(1, Nx+1))
    bbb = list(range(1, Ny+1))
    [m, h1] = meshgrid(aaa)
    [h2, n] = meshgrid(bbb)

    if Ny <= Nx:
        for i in range(Nx - Ny + 1):
            h2[Ny + i - 1,:]=h2[Ny - 1,:]
        h11 = h1[:,0: Ny]
    else:
        h11 = h1


    h = np.exp(1j* np.pi*(h11 + h2))
    FT = np.fft.fft2(h*inData)
    out = h*FT
    return out

def crops(inData,N0,M):
    out = np.zeros((N0, N0),inData.dtype)
    for ii in range(N0):
        for jj in range(N0):
            out[ii, jj] = inData[int(ii + np.floor((M - N0) / 2)),int(jj + np.floor((M - N0) / 2))]
    return out

def supportPro(N=512):
    out = np.zeros((N, N),)
    u = int(np.floor(N / 2))
    v = int(np.floor(N / 2))
    out[u-14-1:u + 14, v-14-1:v+14]=1
    return out

def IFT(inData):
    [Nx,Ny] = inData.shape
    h = np.zeros((Nx, Ny),np.complex)
    aaa = list(range(1, Nx + 1))
    bbb = list(range(1, Ny + 1))
    [m, h1] = meshgrid(aaa)
    [h2, n] = meshgrid(bbb)

    if Ny < Nx:
        for i in range(Nx - Ny + 1):
            h2[Ny + i - 1, :] = h2[Ny - 1, :]
        h11 = h1[:, 0: Ny]
    else:
        h11 = h1

    h = np.exp(-1j * np.pi*(h11 + h2))
    FT2 = np.fft.ifft2(h*inData)
    out = h*FT2
    return out
