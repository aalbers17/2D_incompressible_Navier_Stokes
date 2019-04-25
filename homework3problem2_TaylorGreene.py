import os
import sys
import numpy as np
import scipy.sparse as scysparse
from pdb import set_trace as keyboard
from time import sleep
# import spatial_operators
import spatial_operators
import spatial_operators_2
import scipy.sparse as scysparse
import flowfun as ff
from scipy.optimize import fsolve
import scipy.sparse.linalg as spysparselinalg  # sparse linear algebra
import scipy.linalg as scylinalg               # non-sparse linear algebra
import pylab as plt
from matplotlib import rc as matplotlibrc
from matplotlib import animation
import time # has the equivalent of tic/toc

machine_epsilon = np.finfo(float).eps

#### Runge Kutta Function: ### -- https://www.codeproject.com/Tips/792927/Fourth-Order-Runge-Kutta-Method-in-Python
# inputs: array x, array f(x)
def rk2(a, b, duudx_star,duvdy_star,duvdx_star,dvvdy_star,DivGrad_ustar,DivGrad_vstar,viscu,hs,qu,qv): # 2D 4th order RUnge Kutta
    u_P = -np.ones((Nyc+2,Nxc+2))*np.nan
    v_P = -np.ones((Nyc+2,Nxc+2))*np.nan
    u_P[:,1:-1] = (a[:,1:]+a[:,:-1])/2.
    u_P[:,0] = u_P[:,-2]
    u_P[:,-1] = u_P[:,1]

    u_P[0,:] = u_P[-2,:]
    u_P[-1,:] = u_P[1,:]
    v_P[1:-1,:] = (b[1:,:]+b[:-1,:])/2.
    v_P[:,0] = v_P[:,-2]
    v_P[:,-1] = v_P[:,1]
    v_P[0,:] = v_P[-2,:]
    v_P[-1,:] = v_P[1,:]

    a1 = Ru_num(u_P,v_P,duudx_star,duvdy_star,DivGrad_ustar,viscu)*hs
    b1 = Rv_num(u_P,v_P,duvdx_star,dvvdy_star,DivGrad_vstar,viscv)*hs

    # c1 = fc(a, b, c)*hs
    ak = u_P + a1*0.5
    ak[0,:] = ak[-2,:]
    ak[-1,:] = ak[1,:]
    ak[:,0] = ak[:,-2]
    ak[:,-1] = ak[:,1]

    bk = v_P + b1*0.5
    bk[0,:] = bk[-2,:]
    bk[-1,:] = bk[1,:]
    bk[:,0] = bk[:,-2]
    bk[:,-1] = bk[:,1]
    # print ak.shape
    # ck = c + c1*0.5
    a2 = Ru_num(ak,bk,duudx_star,duvdy_star,DivGrad_ustar,viscu)*hs
    b2 = Rv_num(ak,bk,duvdx_star,dvvdy_star,DivGrad_vstar,viscv)*hs
    # c2 = fc(ak, bk, ck)*hs
    ak = u_P + a2*0.5
    ak[0,:] = ak[-2,:]
    ak[-1,:] = ak[1,:]
    ak[:,0] = ak[:,-2]
    ak[:,-1] = ak[:,1]

    bk = v_P + b2*0.5
    bk[0,:] = bk[-2,:]
    bk[-1,:] = bk[1,:]
    bk[:,0] = bk[:,-2]
    bk[:,-1] = bk[:,1]
    # ck = c + c2*0.5
    a3 = Ru_num(ak,bk,duudx_star,duvdy_star,DivGrad_ustar,viscu)*hs
    b3 = Rv_num(ak,bk,duvdx_star,dvvdy_star,DivGrad_vstar,viscv)*hs
    # c3 = fc(ak, bk, ck)*hs
    ak = u_P + a3
    ak[0,:] = ak[-2,:]
    ak[-1,:] = ak[1,:]
    ak[:,0] = ak[:,-2]
    ak[:,-1] = ak[:,1]

    bk = v_P + b3
    bk[0,:] = bk[-2,:]
    bk[-1,:] = bk[1,:]
    bk[:,0] = bk[:,-2]
    bk[:,-1] = bk[:,1]
    # ck = c + c3
    a4 = Ru_num(ak,bk,duudx_star,duvdy_star,DivGrad_ustar,viscu)*hs
    b4 = Rv_num(ak,bk,duvdx_star,dvvdy_star,DivGrad_vstar,viscv)*hs
    # c4 = fc(ak, bk, ck)*hs
    a = u_P + (a1 + 2.*(a2 + a3) + a4)/6. + qu
    b = v_P + (b1 + 2.*(b2 + b3) + b4)/6. + qv
    a[0,:] = a[-2,:]
    a[-1,:] = a[1,:]
    a[:,0] = a[:,-2]
    a[:,-1] = a[:,1]

    b[0,:] = b[-2,:]
    b[-1,:] = b[1,:]
    b[:,0] = b[:,-2]
    b[:,-1] = b[:,1]
    # c = c + (c1 + 2*(c2 + c3) + c4)/6
    return a, b

def rk2_2(a, b, duudx_star,duvdy_star,duvdx_star,dvvdy_star,DivGrad_ustar,DivGrad_vstar,viscu,hs,qu,qv): # 2D 2nd order RUnge Kutta
    u_P = -np.ones((Nyc+2,Nxc+2))*np.nan
    v_P = -np.ones((Nyc+2,Nxc+2))*np.nan
    u_P[:,1:-1] = (a[:,1:]+a[:,:-1])/2.
    u_P[:,0] = u_P[:,-2]
    u_P[:,-1] = u_P[:,1]

    u_P[0,:] = u_P[-2,:]
    u_P[-1,:] = u_P[1,:]
    v_P[1:-1,:] = (b[1:,:]+b[:-1,:])/2.
    v_P[:,0] = v_P[:,-2]
    v_P[:,-1] = v_P[:,1]
    v_P[0,:] = v_P[-2,:]
    v_P[-1,:] = v_P[1,:]

    a1 = Ru_num(u_P,v_P,duudx_star,duvdy_star,DivGrad_ustar,viscu)*hs
    b1 = Rv_num(u_P,v_P,duvdx_star,dvvdy_star,DivGrad_vstar,viscv)*hs

    # c1 = fc(a, b, c)*hs
    # ak = u_P + a1*0.5
    # ak[0,:] = ak[-2,:]
    # ak[-1,:] = ak[1,:]
    # ak[:,0] = ak[:,-2]
    # ak[:,-1] = ak[:,1]
    #
    # bk = v_P + b1*0.5
    # bk[0,:] = bk[-2,:]
    # bk[-1,:] = bk[1,:]
    # bk[:,0] = bk[:,-2]
    # bk[:,-1] = bk[:,1]
    # # print ak.shape
    # # ck = c + c1*0.5
    # a2 = Ru_num(ak,bk,duudx_star,duvdy_star,DivGrad_ustar,viscu)*hs
    # b2 = Rv_num(ak,bk,duvdx_star,dvvdy_star,DivGrad_vstar,viscv)*hs
    # c2 = fc(ak, bk, ck)*hs
    a = u_P + a1
    a[0,:] = a[-2,:]
    a[-1,:] = a[1,:]
    a[:,0] = a[:,-2]
    a[:,-1] = a[:,1]

    b = v_P + b1
    b[0,:] = b[-2,:]
    b[-1,:] = b[1,:]
    b[:,0] = b[:,-2]
    b[:,-1] = b[:,1]
    # ck = c + c2*0.5
    # a3 = Ru_num(ak,bk,duudx_star,duvdy_star,DivGrad_ustar,viscu)*hs
    # b3 = Rv_num(ak,bk,duvdx_star,dvvdy_star,DivGrad_vstar,viscv)*hs
    # # c3 = fc(ak, bk, ck)*hs
    # ak = u_P + a3
    # ak[0,:] = ak[-2,:]
    # ak[-1,:] = ak[1,:]
    # ak[:,0] = ak[:,-2]
    # ak[:,-1] = ak[:,1]
    #
    # bk = v_P + b3
    # bk[0,:] = bk[-2,:]
    # bk[-1,:] = bk[1,:]
    # bk[:,0] = bk[:,-2]
    # bk[:,-1] = bk[:,1]
    # # ck = c + c3
    # a4 = Ru_num(ak,bk,duudx_star,duvdy_star,DivGrad_ustar,viscu)*hs
    # b4 = Rv_num(ak,bk,duvdx_star,dvvdy_star,DivGrad_vstar,viscv)*hs
    # # c4 = fc(ak, bk, ck)*hs
    # a = u_P + (a1 + 2.*(a2 + a3) + a4)/6. + qu
    # b = v_P + (b1 + 2.*(b2 + b3) + b4)/6. + qv
    # a[0,:] = a[-2,:]
    # a[-1,:] = a[1,:]
    # a[:,0] = a[:,-2]
    # a[:,-1] = a[:,1]
    #
    # b[0,:] = b[-2,:]
    # b[-1,:] = b[1,:]
    # b[:,0] = b[:,-2]
    # b[:,-1] = b[:,1]
    # c = c + (c1 + 2*(c2 + c3) + c4)/6
    return a, b

    # a1 = Ru_num(a,b,duudx_star,duvdy_star,DivGrad_ustar,viscu)*hs
    # b1 = Rv_num(a,b,duvdx_star,dvvdy_star,DivGrad_vstar,viscv)*hs
    # # c1 = fc(a, b, c)*hs
    # ak = a[1:-1,1:-1] + a1*0.5
    # bk = b[1:-1,1:-1] + b1*0.5
    # # ck = c + c1*0.5
    # a2 = Ru_num(ak,bk,duudx_star,duvdy_star,DivGrad_ustar,viscu)*hs
    # b2 = Rv_num(ak,bk,duvdx_star,dvvdy_star,DivGrad_vstar,viscv)*hs
    # # c2 = fc(ak, bk, ck)*hs
    # a = a[1:-1,1:-1] + a2 + qu
    # b = b[1:-1,1:-1] + b2 + qv
    # # c = c + (c1 + 2*(c2 + c3) + c4)/6
    # return a, b

def Ru_num(a,b,duudx_star,duvdy_star,DivGrad_ustar,viscu):
    # print u_n.shape
    # print dudx_star.shape
    # u_P = -np.ones((Nyc+2,Nxc+2))*np.nan
    # v_P = -np.ones((Nyc+2,Nxc+2))*np.nan
    # u_P[:,1:-1] = (u_n[:,1:]+u_n[:,:-1])/2.
    # u_P[:,0] = u_P[:,-2]
    # u_P[:,-1] = u_P[:,1]
    #
    # u_P[0,:] = u_P[-2,:]
    # u_P[-1,:] = u_P[1,:]
    # v_P[1:-1,:] = (v_n[1:,:]+v_n[:-1,:])/2.
    # v_P[:,0] = v_P[:,-2]
    # v_P[:,-1] = v_P[:,1]
    # v_P[0,:] = v_P[-2,:]
    # v_P[-1,:] = v_P[1,:]
    # print a.shape
    u_n[:,:] = (a[:,1:]+a[:,:-1])/2.
    # u_n[:,0] = u_n[:,-1]
    v_n[:,:] = (b[1:,:]+b[:-1,:])/2.
    # v_n[0,:] = v_n[-1,:]

    uu_n = (np.multiply(u_n,u_n))
    vv_n = (np.multiply(v_n,v_n))
    uv_P = (np.multiply(u_P,v_P))
    uv_n = np.multiply((u_n[1:,:]+u_n[:-1,:])/2.,(v_n[:,1:]+v_n[:,:-1])/2.)

    dudx_star[:,1:-1] = (u_n[:,1:]-u_n[:,:-1])/dxc[0]
    dudx_star[:,0] = (u_n[:,0]-u_n[:,-2])/dxc[0]
    dudx_star[:,-1] = (u_n[:,1]-u_n[:,-1])/dxc[0]
    duudx_star[:,1:-1] = (uu_n[:,1:]-uu_n[:,:-1])/dxc[0]

    duvdy_star[1:-1,1:-1] = (uv_P[2:,1:-1]-uv_P[:-2,1:-1])/dyc[0]

    duvdx_star[1:-1,1:-1] = (uv_P[1:-1:,2:]-uv_P[1:-1,:-2])/dxc[0]


    dudy_star[1:-1,:] = (u_n[2:-1,:]-u_n[1:-2,:])/dyc[0]
    dudy_star[0,:] = (u_n[1,:]-u_n[0,:])/dyc[0]
    dudy_star[-1,:] = (u_n[-1,:]-u_n[-2,:])/dyc[0]

    dvdx_star[:,1:-1] = (v_n[:,2:-1]-v_n[:,1:-2])/dxc[0]
    dvdx_star[:,0] = (v_n[:,1]-v_n[:,0])/dxc[0]
    dvdx_star[:,-1] = (v_n[:,-1]-v_n[:,-2])/dxc[0]

    dvdy_star[1:-1,:] = (v_n[1:,:]-v_n[:-1,:])/dyc[0]
    dvdy_star[0,:] = (v_n[0,:]-v_n[-2,:])/dxc[0]
    dvdy_star[-1,:] = (v_n[1,:]-v_n[-1,:])/dxc[0]
    dvvdy_star[1:-1,:] = (vv_n[1:,:]-vv_n[:-1,:])/dyc[0]

    dpdx_star[:,1:-1] = (p_n[:,2:-1]-p_n[:,1:-2])/dxc[0]
    dpdx_star[0,:] = 0.
    dpdx_star[-1,:] = 0.
    dpdx_star[:,0] = 0.
    dpdx_star[:,-1] = 0.

    dpdy_star[1:-1,:] = (p_n[2:-1,:]-p_n[1:-2,:])/dyc[0]
    dpdy_star[0,:] = 0.
    dpdy_star[-1,:] = 0.
    dpdy_star[:,0] = 0.
    dpdy_star[:,-1] = 0.

    DivGrad_ustar[1:-1,1:-1] = 1./Dxc[1:-1,1:-1]*((u_P[1:-1,2:]-u_P[1:-1,1:-1])/Dxc[1:-1,2:] - (u_P[1:-1,1:-1]-u_P[1:-1,:-2])/Dxc[1:-1,:-2]) + 1./Dyc[1:-1,1:-1]*((u_P[2:,1:-1]-u_P[1:-1,1:-1])/Dyc[2:,1:-1] - (u_P[1:-1,1:-1]-u_P[:-2,1:-1])/Dyc[:-2,1:-1])

    DivGrad_vstar[1:-1,1:-1] = 1./Dxc[1:-1,1:-1]*((v_P[1:-1,2:]-v_P[1:-1,1:-1])/Dxc[1:-1,2:] - (v_P[1:-1,1:-1]-v_P[1:-1,:-2])/Dxc[1:-1,:-2]) + 1./Dyc[1:-1,1:-1]*((v_P[2:,1:-1]-v_P[1:-1,1:-1])/Dyc[2:,1:-1] - (v_P[1:-1,1:-1]-v_P[:-2,1:-1])/Dyc[:-2,1:-1])


    Ruc = -(duudx_star + duvdy_star)

    Ruv = (viscu*DivGrad_ustar)

    return Ruc + Ruv

def Rv_num(a,b,duvdx_star,dvvdy_star,DivGrad_vstar,viscu):

    # u_P = -np.ones((Nyc+2,Nxc+2))*np.nan
    # v_P = -np.ones((Nyc+2,Nxc+2))*np.nan
    # u_P[:,1:-1] = (u_n[:,1:]+u_n[:,:-1])/2.
    # u_P[:,0] = u_P[:,-2]
    # u_P[:,-1] = u_P[:,1]
    #
    # u_P[0,:] = u_P[-2,:]
    # u_P[-1,:] = u_P[1,:]
    # v_P[1:-1,:] = (v_n[1:,:]+v_n[:-1,:])/2.
    # v_P[:,0] = v_P[:,-2]
    # v_P[:,-1] = v_P[:,1]
    # v_P[0,:] = v_P[-2,:]
    # v_P[-1,:] = v_P[1,:]
    u_n[:,:] = (a[:,1:]+a[:,:-1])/2.
    # u_n[:,0] = u_n[:,-1]
    v_n[:,:] = (b[1:,:]+b[:-1,:])/2.
    # v_n[0,:] = v_n[-1,:]

    uu_n = (np.multiply(u_n,u_n))
    vv_n = (np.multiply(v_n,v_n))
    uv_P = (np.multiply(u_P,v_P))
    uv_n = np.multiply((u_n[1:,:]+u_n[:-1,:])/2.,(v_n[:,1:]+v_n[:,:-1])/2.)

    dudx_star[:,1:-1] = (u_n[:,1:]-u_n[:,:-1])/dxc[0]
    dudx_star[:,0] = (u_n[:,0]-u_n[:,-2])/dxc[0]
    dudx_star[:,-1] = (u_n[:,1]-u_n[:,-1])/dxc[0]
    duudx_star[:,1:-1] = (uu_n[:,1:]-uu_n[:,:-1])/dxc[0]

    duvdy_star[1:-1,1:-1] = (uv_P[2:,1:-1]-uv_P[:-2,1:-1])/dyc[0]

    duvdx_star[1:-1,1:-1] = (uv_P[1:-1:,2:]-uv_P[1:-1,:-2])/dxc[0]


    dudy_star[1:-1,:] = (u_n[2:-1,:]-u_n[1:-2,:])/dyc[0]
    dudy_star[0,:] = (u_n[1,:]-u_n[0,:])/dyc[0]
    dudy_star[-1,:] = (u_n[-1,:]-u_n[-2,:])/dyc[0]

    dvdx_star[:,1:-1] = (v_n[:,2:-1]-v_n[:,1:-2])/dxc[0]
    dvdx_star[:,0] = (v_n[:,1]-v_n[:,0])/dxc[0]
    dvdx_star[:,-1] = (v_n[:,-1]-v_n[:,-2])/dxc[0]

    dvdy_star[1:-1,:] = (v_n[1:,:]-v_n[:-1,:])/dyc[0]
    dvdy_star[0,:] = (v_n[0,:]-v_n[-2,:])/dxc[0]
    dvdy_star[-1,:] = (v_n[1,:]-v_n[-1,:])/dxc[0]
    dvvdy_star[1:-1,:] = (vv_n[1:,:]-vv_n[:-1,:])/dyc[0]

    dpdx_star[:,1:-1] = (p_n[:,2:-1]-p_n[:,1:-2])/dxc[0]
    dpdx_star[0,:] = 0.
    dpdx_star[-1,:] = 0.
    dpdx_star[:,0] = 0.
    dpdx_star[:,-1] = 0.

    dpdy_star[1:-1,:] = (p_n[2:-1,:]-p_n[1:-2,:])/dyc[0]
    dpdy_star[0,:] = 0.
    dpdy_star[-1,:] = 0.
    dpdy_star[:,0] = 0.
    dpdy_star[:,-1] = 0.

    DivGrad_ustar[1:-1,1:-1] = 1./Dxc[1:-1,1:-1]*((u_P[1:-1,2:]-u_P[1:-1,1:-1])/Dxc[1:-1,2:] - (u_P[1:-1,1:-1]-u_P[1:-1,:-2])/Dxc[1:-1,:-2]) + 1./Dyc[1:-1,1:-1]*((u_P[2:,1:-1]-u_P[1:-1,1:-1])/Dyc[2:,1:-1] - (u_P[1:-1,1:-1]-u_P[:-2,1:-1])/Dyc[:-2,1:-1])

    DivGrad_vstar[1:-1,1:-1] = 1./Dxc[1:-1,1:-1]*((v_P[1:-1,2:]-v_P[1:-1,1:-1])/Dxc[1:-1,2:] - (v_P[1:-1,1:-1]-v_P[1:-1,:-2])/Dxc[1:-1,:-2]) + 1./Dyc[1:-1,1:-1]*((v_P[2:,1:-1]-v_P[1:-1,1:-1])/Dyc[2:,1:-1] - (v_P[1:-1,1:-1]-v_P[:-2,1:-1])/Dyc[:-2,1:-1])



    Rvc = -(duvdx_star +dvvdy_star)
    Rvv = (viscu*DivGrad_vstar)

    return Rvc + Rvv
#########################################
############### User Input ##############

# number of (pressure) cells = mass conservation cells
# foundn optimal w's: Re = 10**-2, w = 1.40, iter = 33; Re = 10**0, w = 1.40, iter = 33 ; Re = 10**2, w = 1.40 iter = 19; Re = 10**4, w = 1.40, iter =
# 758 1.5
# Nx = np.array([30, 48, 60, 72, 84, 96])
# Ny = np.array([30, 48, 60, 72, 84, 96])
Nx = np.array([24,24,48,64])
Ny = np.array([24,24,48,64])
#
rmsu_check = np.zeros(Nx.size,)
rmsu = np.zeros(Nx.size,)
rmsu2 = np.zeros(Nx.size,)
rmsv = np.zeros(Ny.size,)
rmsv2 = np.zeros(Ny.size,)
rmsv_check = np.zeros(Nx.size,)
order_accuracy = np.zeros(Nx.size,)
h = np.zeros(Nx.size,)
# # Rv_check = np.zeros(Nx.size,)
# # Rv_numerical = np.zeros(Nx.size,)

# # Nxc  = np.array([6, 12, 24, 48, 96])
# # Nyc  = np.array([6,12,24,48,96])
for i in range(Nx.size):
    Nxc = Nx[i]
    Nyc = Ny[i]
    Np   = Nxc*Nyc

# Nu = (Nxc-1)*Nyc
# Nv = Nxc*(Nyc-1)
    Nt = np.array([100, 100, 100])
    # Nt = 100
    dt = np.zeros(Nt.size,)
    rms4 = np.zeros(Nt.size,)
    # rmsu_check = np.zeros(Nx.size,)
    Lx   = 2.*np.pi
    Ly   = 2.*np.pi
    Re = 10.**0
    visc = 1.0
    w = 1.4 # relaxation parameter
    # b = 1.0
    # n = 3
    # n = np.array([1., np.float64(Nxc/2.), np.float64(Nxc)])

    #########################################
    ######## Preprocessing Stage ############

# for i in range(Nt.size):
    # define grid for u and v velocity components first
    # and then define pressure cells locations
    xsi_u = np.linspace(0.,1.0,Nxc+1)
    xsi_v = np.linspace(0.,1.0,Nyc+1)
    xplot = np.linspace(0.,1.0,Nxc)
    yplot = np.linspace(0.,1.0,Nyc)
    t = np.linspace(0.,1.0,Nt[0])
    dt[0] = t[1]-t[0]
    # uniform grid
    xu = (xsi_u)*Lx
    xu_u = xu # separating u cell calc from pressure cell calc
    xu_ddxu = xu[1:]
    yv = (xsi_v)*Ly
    yv_v = yv # separating v cell calc from pressure cell calc
    yv_ddyv = yv[1:]

    # creating ghost cells
    dxu0 = np.diff(xu)[0]
    dxuL = np.diff(xu)[-1]
    xu = np.concatenate([[xu[0]-dxu0],xu,[xu[-1]+dxuL]])
    # xu_u = np.concatenate([xu_u,[xu_u[-1]+dxuL]])
    dyv0 = np.diff(yv)[0]
    dyvL = np.diff(yv)[-1]
    yv = np.concatenate([[yv[0]-dyv0],yv,[yv[-1]+dyvL]])
    # yv_v = np.concatenate([yv_v,[yv_v[-1]+dyvL]])
    dxc = np.diff(xu)  # pressure-cells spacings
    dyc = np.diff(yv)

    xc = 0.5*(xu[:-1]+xu[1:])  # total of Nxc cells
    yc = 0.5*(yv[:-1]+yv[1:])  # total of Nyc cells

    # note that indexing is Xc[j_y,i_x] or Xc[j,i]
    [Xc,Yc]   = np.meshgrid(xc,yc)     # these arrays are with ghost cells
    [Xplot,Yplot] = np.meshgrid(xplot,yplot)
    [Dxc,Dyc] = np.meshgrid(dxc,dyc)   # these arrays are with ghost cells
    [Xu, Yu] = np.meshgrid(xu_u,yc) # u velocity grid with ghost cells
    [Xv, Yv] = np.meshgrid(xc,yv_v) # v velocity grid with ghost cells
    [Xuv,Yuv] = np.meshgrid(xu_u,yv_v)
    [Xu_DDxU,Yu_DDxU] = np.meshgrid(xu_ddxu,yc)
    [Xv_DDyV, Yv_DDyV] = np.meshgrid(xc,yv_ddyv)

    # print Xu
    # print Yv
    # Pre-allocated at all False = no fluid points
    pressureCells_Mask = np.zeros(Xc.shape)
    pressureCells_Mask[1:-1,1:-1] = True

    # Pre-allocated at all False = no fluid points
    uCells_Mask = np.zeros(Xu.shape)
    uCells_internal_Mask = np.zeros(Xu.shape)
    uCells_Mask[1:-1,:] = True

    vCells_Mask = np.zeros(Yv.shape)
    vCells_internal_Mask = np.zeros(Yv.shape)
    vCells_Mask[:,1:-1] = True

    # vCells_internal_Mask[2:-2,1:-1] = True

    # print vCells_internal_Mask
    # Introducing obstacle in pressure Mask
    obstacle_radius = 0.*Lx # set equal to 0.0*Lx to remove obstacle
    distance_from_center = np.sqrt(np.power(Xc-Lx/2.,2.0)+np.power(Yc-Ly/2.,2.0))
    j_obstacle,i_obstacle = np.where(distance_from_center<obstacle_radius)
    pressureCells_Mask[j_obstacle,i_obstacle] = False

    qp  = np.ones(Np,)

    # a more advanced option is to separately create the divergence and gradient operators
    # DivGrad = spatial_operators.create_DivGrad_operator(Dxc,Dyc,Xc,Yc,pressureCells_Mask,boundary_conditions="Homogeneous Dirichlet")
    DivGradP = spatial_operators.create_DivGrad_operator(Dxc,Dyc,Xc,Yc,pressureCells_Mask,boundary_conditions="Homogeneous Neumann")# "Homogeneous Neumann")
    DivGrad = spatial_operators_2.create_DivGrad_operator(Dxc,Dyc,Xc,Yc,pressureCells_Mask,boundary_conditions="Periodic")
    # DivGradU = spatial_operators_2.create_DivGrad_operator(Dxc,Dyc,Xu,Yc,pressureCells_Mask,uCells_Mask,"Periodic")

    ## initial conditions:
    u_n = -np.ones((Nyc+2,Nxc+1))*np.nan
    u_n2 = -np.ones((Nyc+2,Nxc+1))*np.nan
    ustar = -np.ones((Nyc+2,Nxc+1))*np.nan
    ustar2 = -np.ones((Nyc+2,Nxc+1))*np.nan

    u_n[np.where(uCells_Mask==True)] = -np.cos(Xu[np.where(uCells_Mask==True)])*np.sin(Yc[np.where(uCells_Mask==True)])
    u_n2[np.where(uCells_Mask==True)] = -np.cos(Xu[np.where(uCells_Mask==True)])*np.sin(Yc[np.where(uCells_Mask==True)])
    u_n[0,:] = u_n[-2,:]
    u_n[-1,:] = u_n[1,:]
    u_n2[0,:] = u_n2[-2,:]
    u_n2[-1,:] = u_n2[1,:]

    v_n = -np.ones((Nyc+1,Nxc+2))*np.nan
    vstar = -np.ones((Nyc+1,Nxc+2))*np.nan
    v_n2 = -np.ones((Nyc+1,Nxc+2))*np.nan
    vstar2 = -np.ones((Nyc+1,Nxc+2))*np.nan

    v_n[np.where(vCells_Mask==True)] = np.sin(Xc[np.where(vCells_Mask==True)])*np.cos(Yv[np.where(vCells_Mask==True)])
    v_n2[np.where(vCells_Mask==True)] = np.sin(Xc[np.where(vCells_Mask==True)])*np.cos(Yv[np.where(vCells_Mask==True)])
    v_n[:,0] = v_n[:,-2]
    v_n[:,-1] = v_n[:,1]
    v_n2[:,0] = v_n2[:,-2]
    v_n2[:,-1] = v_n2[:,1]

    a = -np.ones((Nyc+2,Nxc+2))*np.nan
    a1= -np.ones((Nyc+2,Nxc+2))*np.nan
    a2 = -np.ones((Nyc+2,Nxc+2))*np.nan
    b = -np.ones((Nyc+2,Nxc+2))*np.nan
    u_P = -np.ones((Nyc+2,Nxc+2))*np.nan
    v_P = -np.ones((Nyc+2,Nxc+2))*np.nan
    dudx_star = -np.ones((Nyc+2,Nxc+2))*np.nan
    duudx_star = -np.ones((Nyc+2,Nxc+2))*np.nan
    dudy_star = -np.ones((Nyc+1,Nxc+1))*np.nan
    duvdy_star = -np.ones((Nyc+2,Nxc+2))*np.nan
    duvdx_star = -np.ones((Nyc+2,Nxc+2))*np.nan
    dvvdy_star = -np.ones((Nyc+2,Nxc+2))*np.nan
    dvdx_star = -np.ones((Nyc+1,Nxc+1))*np.nan
    dvdy_star = -np.ones((Nyc+2,Nxc+2))*np.nan
    dpdx_star = -np.ones((Nyc+2,Nxc+1))*np.nan
    dpdy_star = -np.ones((Nyc+1,Nxc+2))*np.nan
    DivGrad_ustar = -np.ones((Nyc+2,Nxc+2))*np.nan
    DivGrad_vstar = -np.ones((Nyc+2,Nxc+2))*np.nan
    DivGrad_pstar = -np.ones((Nyc+2,Nxc+2))*np.nan

    # Neumann BCs for Divgrad_pstar:

    p_n = -np.ones((Nyc+2,Nxc+2))*np.nan # initialize initial guess P0
    p_n[np.where(pressureCells_Mask==True)] = -1./4.*(np.cos(2.*Xc[np.where(pressureCells_Mask==True)])+np.cos(2.*Yc[np.where(pressureCells_Mask==True)]))
    p_n[0,:] = p_n[1,:]
    p_n[-1,:] = p_n[-2,:]
    p_n[:,0] = p_n[:,1]
    p_n[:,-1] = p_n[:,-2]

    qu = np.ones((Np,1))
    qv = np.ones((Np,1))
    qp = np.ones((Np,1))
    # qp = np.reshape(p_n[np.where(pressureCells_Mask==True)],(Np,1))


    viscu = 1.#np.ones((Nu,1))
    viscv = 1.

    figwidth       = 10
    figheight      = 6
    lineWidth      = 4
    textFontSize   = 14
    gcafontSize    = 30

    rk = np.zeros(1000,)
    # Runge Kutta Methods:
    #RK1:
    iter = 0
    subiter = 0
    h[i] = np.sqrt(dxu0**2+dyv0**2)
    # for j in range(t.size):

    # solution:
    uanalytical = -np.exp(-2.*t[0])*np.cos(Xc[1:-1,1:-1])*np.sin(Yc[1:-1,1:-1])
    vanalytical = np.exp(-2.*t[0])*np.sin(Xc[1:-1,1:-1])*np.cos(Yc[1:-1,1:-1])
    panalytical = -np.exp(-4.*t[0])/4.*(np.cos(2.*Xc[1:-1,1:-1])+np.cos(2.*Yc[1:-1,1:-1]))# -np.exp(-4.*t[0])/4.*(np.cos(2.*Xc[np.where(pressureCells_Mask==True)])+np.cos(2.*Yc[np.where(pressureCells_Mask==True)]))
    duanalytical_dx =  np.sin(Yc[1:-1,1:-1])*np.sin(Xc[1:-1,1:-1])
    duanalytical_dy = -np.cos(Yc[1:-1,1:-1])*np.cos(Xc[1:-1,1:-1])
    lap_uanalytical = 2*np.cos(Xc[1:-1,1:-1])*np.sin(Yc[1:-1,1:-1])
    lap_vanalytical = -2*np.cos(Xc[1:-1,1:-1])*np.sin(Yc[1:-1,1:-1])
    Ru_check = -(np.multiply(duanalytical_dx,uanalytical) + np.multiply(duanalytical_dy,vanalytical)) + viscu*lap_uanalytical

    t = 0
    dtconv = np.min(dxc[0]/np.linalg.norm(u_n))
    dtvisc = np.min(dxc[0]**2/1.0)
    dtmin = np.hstack([dtconv,dtvisc])
    dt = np.min(dtmin)
    if dt > .01:
        dt = .01
    while t < 1.0: # j in range(t.size):
    # print Xc
        t += dt
        print t
        iter +=1
        print iter
        u_P = -np.ones((Nyc+2,Nxc+2))*np.nan
        v_P = -np.ones((Nyc+2,Nxc+2))*np.nan
        u_P2 = -np.ones((Nyc+2,Nxc+2))*np.nan
        v_P2 = -np.ones((Nyc+2,Nxc+2))*np.nan
        u_P[:,1:-1] = (u_n[:,1:]+u_n[:,:-1])/2.
        u_P[:,0] = u_P[:,-2]
        u_P[:,-1] = u_P[:,1]
        u_P[0,:] = u_P[-2,:]
        u_P[-1,:] = u_P[1,:]

        u_P2[:,1:-1] = (u_n2[:,1:]+u_n2[:,:-1])/2.
        u_P2[:,0] = u_P2[:,-2]
        u_P2[:,-1] = u_P2[:,1]
        u_P2[0,:] = u_P2[-2,:]
        u_P2[-1,:] = u_P2[1,:]

        v_P[1:-1,:] = (v_n[1:,:]+v_n[:-1,:])/2.
        v_P[:,0] = v_P[:,-2]
        v_P[:,-1] = v_P[:,1]
        v_P[0,:] = v_P[-2,:]
        v_P[-1,:] = v_P[1,:]

        v_P2[1:-1,:] = (v_n2[1:,:]+v_n2[:-1,:])/2.
        v_P2[:,0] = v_P2[:,-2]
        v_P2[:,-1] = v_P2[:,1]
        v_P2[0,:] = v_P2[-2,:]
        v_P2[-1,:] = v_P2[1,:]

        uu_n = (np.multiply(u_n,u_n))
        vv_n = (np.multiply(v_n,v_n))
        uv_P = (np.multiply(u_P,v_P))
        uv_n = np.multiply((u_n[1:,:]+u_n[:-1,:])/2.,(v_n[:,1:]+v_n[:,:-1])/2.)

        dudx_star[:,1:-1] = (u_n[:,1:]-u_n[:,:-1])/dxc[0]
        dudx_star[:,0] = (u_n[:,0]-u_n[:,-2])/dxc[0]
        dudx_star[:,-1] = (u_n[:,1]-u_n[:,-1])/dxc[0]
        duudx_star[:,1:-1] = (uu_n[:,1:]-uu_n[:,:-1])/dxc[0]

        duvdy_star[1:-1,1:-1] = (uv_P[2:,1:-1]-uv_P[:-2,1:-1])/dyc[0]

        duvdx_star[1:-1,1:-1] = (uv_P[1:-1:,2:]-uv_P[1:-1,:-2])/dxc[0]


        dudy_star[1:-1,:] = (u_n[2:-1,:]-u_n[1:-2,:])/dyc[0]
        dudy_star[0,:] = (u_n[1,:]-u_n[0,:])/dyc[0]
        dudy_star[-1,:] = (u_n[-1,:]-u_n[-2,:])/dyc[0]

        dvdx_star[:,1:-1] = (v_n[:,2:-1]-v_n[:,1:-2])/dxc[0]
        dvdx_star[:,0] = (v_n[:,1]-v_n[:,0])/dxc[0]
        dvdx_star[:,-1] = (v_n[:,-1]-v_n[:,-2])/dxc[0]

        dvdy_star[1:-1,:] = (v_n[1:,:]-v_n[:-1,:])/dyc[0]
        dvdy_star[0,:] = (v_n[0,:]-v_n[-2,:])/dxc[0]
        dvdy_star[-1,:] = (v_n[1,:]-v_n[-1,:])/dxc[0]
        dvvdy_star[1:-1,:] = (vv_n[1:,:]-vv_n[:-1,:])/dyc[0]

        dpdx_star[:,1:-1] = (p_n[:,2:-1]-p_n[:,1:-2])/dxc[0]
        dpdx_star[0,:] = 0.
        dpdx_star[-1,:] = 0.
        dpdx_star[:,0] = 0.
        dpdx_star[:,-1] = 0.

        dpdy_star[1:-1,:] = (p_n[2:-1,:]-p_n[1:-2,:])/dyc[0]
        dpdy_star[0,:] = 0.
        dpdy_star[-1,:] = 0.
        dpdy_star[:,0] = 0.
        dpdy_star[:,-1] = 0.

        DivGrad_ustar[1:-1,1:-1] = 1./Dxc[1:-1,1:-1]*((u_P[1:-1,2:]-u_P[1:-1,1:-1])/Dxc[1:-1,2:] - (u_P[1:-1,1:-1]-u_P[1:-1,:-2])/Dxc[1:-1,:-2]) + 1./Dyc[1:-1,1:-1]*((u_P[2:,1:-1]-u_P[1:-1,1:-1])/Dyc[2:,1:-1] - (u_P[1:-1,1:-1]-u_P[:-2,1:-1])/Dyc[:-2,1:-1])

        DivGrad_vstar[1:-1,1:-1] = 1./Dxc[1:-1,1:-1]*((v_P[1:-1,2:]-v_P[1:-1,1:-1])/Dxc[1:-1,2:] - (v_P[1:-1,1:-1]-v_P[1:-1,:-2])/Dxc[1:-1,:-2]) + 1./Dyc[1:-1,1:-1]*((v_P[2:,1:-1]-v_P[1:-1,1:-1])/Dyc[2:,1:-1] - (v_P[1:-1,1:-1]-v_P[:-2,1:-1])/Dyc[:-2,1:-1])

        DivGrad_pstar[1:-1,1:-1] = 1./Dxc[1:-1,1:-1]*(dpdx_star[1:-1,1:]-dpdx_star[1:-1,:-1]) + 1./Dyc[1:-1,1:-1]*(dpdy_star[1:,1:-1]-dpdy_star[:-1,1:-1])

        # rmsu_check[i] = np.sqrt(np.mean((DivGrad_ustar[1:-1,1:-1]-lap_uanalytical)**2))
        # rmsv_check[i] = np.sqrt(np.mean((DivGrad_vstar[1:-1,1:-1]-lap_vanalytical)**2))
        # break;
        qp = np.zeros((Nyc,Nxc))
        qu = 0.
        qv = 0.

        uanalytical = -np.exp(-2.*t)*np.cos(Xc[1:-1,1:-1])*np.sin(Yc[1:-1,1:-1])
        vanalytical = np.exp(-2.*t)*np.sin(Xc[1:-1,1:-1])*np.cos(Yc[1:-1,1:-1])
        panalytical = -np.exp(-4.*t)/4.*(np.cos(2.*Xc[1:-1,1:-1])+np.cos(2.*Yc[1:-1,1:-1]))# -np.exp(-4.*t[0])/4.*(np.cos(2.*Xc[np.where(pressureCells_Mask==True)])+np.cos(2.*Yc[np.where(pressureCells_Mask==True)]))



        a, b = rk2(u_n, v_n, duudx_star,duvdy_star,duvdx_star,dvvdy_star,DivGrad_ustar,DivGrad_vstar,viscu,dt,qu,qv)
        a2, b2 = rk2_2(u_n2, v_n2, duudx_star,duvdy_star,duvdx_star,dvvdy_star,DivGrad_ustar,DivGrad_vstar,viscu,dt,qu,qv)

        # print a
        # plt.contourf(a)
        # plt.show()
        a[:,-1] = a[:,1]
        a[:,0] = a[:,-2]
        b[-1,:] = b[1,:]
        b[0,:] = b[-2,:]
        a2[:,-1] = a2[:,1]
        a2[:,0] = a2[:,-2]
        b2[-1,:] = b2[1,:]
        b2[0,:] = b2[-2,:]
        ustar[:,:] = (a[:,1:]+a[:,:-1])/2.
        vstar[:,:] = (b[1:,:]+b[:-1,:])/2.

        ustar2[:,:] = (a2[:,1:]+a2[:,:-1])/2.
        vstar2[:,:] = (b2[1:,:]+b2[:-1,:])/2.

        Dx_ustar = (ustar[:,1:]-ustar[:,:-1])/dxc[0]
        Dy_vstar = (vstar[1:,:]-vstar[:-1,:])/dyc[0]

        Dx_ustar2 = (ustar2[:,1:]-ustar2[:,:-1])/dxc[0]
        Dy_vstar2 = (vstar2[1:,:]-vstar2[:-1,:])/dyc[0]

        r0 = np.linalg.norm(Dx_ustar[1:-1,:]+Dy_vstar[:,1:-1])
        rk = r0
        aa = 2.*dt*(1./Dxc[1:-1,1:-1]**2+1./Dyc[1:-1,1:-1]**2)
        bb = -dt/Dxc[1:-1,1:-1]**2
        cc = -dt/Dyc[1:-1,1:-1]**2

        subiter = 0
        while rk/r0 > .00001:
            # print rk/r0
            subiter += 1
            Dx_ustar = (ustar[:,1:]-ustar[:,:-1])/dxc[0]
            Dy_vstar = (vstar[1:,:]-vstar[:-1,:])/dyc[0]

            p_n[1:-1,1:-1] = -1./aa[:,:]*(bb[:,:]*(p_n[1:-1,2:]+p_n[1:-1,:-2]) + cc[:,:]*(p_n[2:,1:-1]+p_n[:-2,1:-1])  -qp + Dx_ustar[1:-1,:] + Dy_vstar[:,1:-1])
            # p_n[1:-1,1:-1] = np.reshape(spysparselinalg.spsolve(DivGradP,1/dt[0]*(-qp.flatten() + Dx_ustar[1:-1,:].flatten()+Dy_vstar[:,1:-1].flatten())),(Nyc,Nxc))

            p_n[1:-1,0] = p_n[1:-1,1]
            p_n[1:-1,-1] = p_n[1:-1,-2]
            p_n[0,1:-1] = p_n[1,1:-1]
            p_n[-1,1:-1] = p_n[-2,1:-1]

            ustar[:,:] = ustar[:,:] - dt/Dxc[:,:-1]*(p_n[:,1:] - p_n[:,:-1])
            vstar[:,:] = vstar[:,:] - dt/Dyc[:-1,:]*(p_n[1:,:]-p_n[:-1,:])


            qp = (ustar[1:-1,1:]-ustar[1:-1,:-1])/dxc[0] + (vstar[1:,1:-1]-vstar[:-1,1:-1])/dyc[0]

            rk = np.linalg.norm(-qp+Dx_ustar[1:-1,:]+Dy_vstar[:,1:-1])
                # print subiter
            # u_correct = 1./dt[0]/Dxc[:,1:-1]*(p_n[])
        # print iter
        r0 = np.linalg.norm(Dx_ustar2[1:-1,:]+Dy_vstar2[:,1:-1])
        rk = r0
        aa = 2.*dt*(1./Dxc[1:-1,1:-1]**2+1./Dyc[1:-1,1:-1]**2)
        bb = -dt/Dxc[1:-1,1:-1]**2
        cc = -dt/Dyc[1:-1,1:-1]**2

        subiter = 0
        while rk/r0 > .00001:
            # print rk/r0
            subiter += 1
            Dx_ustar2 = (ustar2[:,1:]-ustar2[:,:-1])/dxc[0]
            Dy_vstar2 = (vstar2[1:,:]-vstar2[:-1,:])/dyc[0]

            p_n[1:-1,1:-1] = -1./aa[:,:]*(bb[:,:]*(p_n[1:-1,2:]+p_n[1:-1,:-2]) + cc[:,:]*(p_n[2:,1:-1]+p_n[:-2,1:-1])  -qp + Dx_ustar2[1:-1,:] + Dy_vstar2[:,1:-1])
            # p_n[1:-1,1:-1] = np.reshape(spysparselinalg.spsolve(DivGradP,1/dt[0]*(-qp.flatten() + Dx_ustar[1:-1,:].flatten()+Dy_vstar[:,1:-1].flatten())),(Nyc,Nxc))

            p_n[1:-1,0] = p_n[1:-1,1]
            p_n[1:-1,-1] = p_n[1:-1,-2]
            p_n[0,1:-1] = p_n[1,1:-1]
            p_n[-1,1:-1] = p_n[-2,1:-1]

            ustar2[:,:] = ustar2[:,:] - dt/Dxc[:,:-1]*(p_n[:,1:] - p_n[:,:-1])
            vstar2[:,:] = vstar2[:,:] - dt/Dyc[:-1,:]*(p_n[1:,:]-p_n[:-1,:])


            qp = (ustar2[1:-1,1:]-ustar2[1:-1,:-1])/dxc[0] + (vstar2[1:,1:-1]-vstar2[:-1,1:-1])/dyc[0]

            rk = np.linalg.norm(-qp+Dx_ustar2[1:-1,:]+Dy_vstar2[:,1:-1])

        u_n = ustar
        v_n = vstar

        u_n2 = ustar2
        v_n2 = vstar2

        mag_vel = .5*(u_P**2 + v_P**2)
        psi = ff.psi_transform(Xc,Yc,Dxc,Dyc,u_P,v_P)
        # plt.contourf(psi)
        # plt.draw()
        # plt.colorbar()
        # plt.pause(.2)
        # plt.gcf().clear()
        # print uanalytical
        # widths = np.linspace(0, 2, Xc.size)

        plt.quiver(Xc,Yc,u_P,v_P,scale = 15)
        # plt.quiver(Xc,Yc,u_P,v_P,scale=15)
        # # plt.contourf(p_n)
        plt.draw()
        plt.pause(.2)
        plt.gcf().clear()

    # plt.quiver(Xc,Yc,u_P,v_P,scale = 15)
    # plt.show()
#     rmsu[i] = np.sqrt(np.mean((u_P[1:-1,1:-1]-uanalytical)**2))
#     rmsu2[i] = np.sqrt(np.mean((u_P2[1:-1,1:-1]-uanalytical)**2))
#     rmsv[i] = np.sqrt(np.mean((v_P[1:-1:,1:-1]-vanalytical)**2))
#     rmsv[i] = np.sqrt(np.mean((v_P2[1:-1:,1:-1]-vanalytical)**2))
#     order_accuracy[i] = (Nxc/Lx)**-2
# figure_name = "DivGrad of u,v check"
# # plt.loglog(1/h,rmsu_check,label='DivGrad rmsu vs. Laplacian')
# plt.loglog(1/h,rmsu,label='RK4 rmsu vs. Laplacian')
# plt.loglog(1/h,rmsv,label='RK4 rmsv vs. vhat')
# plt.loglog(1/h,rmsu2,label='RK1 rmsu u vs. uhat')
# plt.loglog(1/h,rmsv2,label='RK1 rmsv vs. vhat')
# plt.loglog(1/h,order_accuracy,label='2nd order accuracy')
# # plt.plot(list(range(Np)),(uanalytical-u_n_calc))
# # plt.plot(list(range(iter1)),rk1,label='N = 10^0')
# # plt.plot(list(range(iter2)),rk2[:iter2],label='N =10^2')
# # plt.plot(list(range(iter3)),rk3[:iter3],label='N =10^4')
# # plt.plot(list(range(iter4)),rk4[:iter4],label='N =10^6')
# # plt.plot(list(range(iter5)),rk5[:iter5],label='N =10^8')
# plt.xlabel("1/h",fontsize=textFontSize)
# plt.ylabel("RMS",fontsize=textFontSize,rotation=90)
# plt.grid('on',which='both')
# plt.title("1/h vs. RMS ")
# plt.legend()
# plt.tight_layout()
# plt.show()
# plt.savefig(figure_name)
# # plt.pause(.2)
# quit()
        # plt.clf()
        # plt.cla()
        # plt.close()
