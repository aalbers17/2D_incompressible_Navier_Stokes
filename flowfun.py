from scipy import integrate
import numpy
from scipy import optimize
from scipy.signal import fftconvolve
import pylab as plt
# this method was found from: https://stackoverflow.com/questions/49557329/compute-stream-function-from-x-and-y-velocities-by-integration-in-python
# function convert u,v, to psi and vice versa
# because of intergration error, a minimization function is utilized to reduce this error accumulation
# Credit: https://stackoverflow.com/users/2005415/jason
#     return x**2
def uRecon(sf,vp,kernel_x,kernel_y):
    uchi=fftconvolve(vp,-kernel_x,mode='valid')
    upsi=fftconvolve(sf,kernel_y,mode='valid')
    return upsi+uchi

def vRecon(sf,vp,kernel_x,kernel_y):
    vchi=fftconvolve(vp,-kernel_y,mode='valid')
    vpsi=fftconvolve(sf,-kernel_x,mode='valid')
    return vpsi+vchi

def costFunc(params,u,v,kernel_x,kernel_y,pad_shape,lam):
    pp=params.reshape(pad_shape)
    sf=pp[0]
    vp=pp[1]
    uhat=uRecon(sf,vp,kernel_x,kernel_y)
    vhat=vRecon(sf,vp,kernel_x,kernel_y)
    j=(uhat-u)**2+(vhat-v)**2
    j=j.mean()
    j+=lam*numpy.mean(params**2)

    return j

def jac(params,u,v,kernel_x,kernel_y,pad_shape,lam):
    pp=params.reshape(pad_shape)
    sf=pp[0]
    vp=pp[1]
    uhat=uRecon(sf,vp,kernel_x,kernel_y)
    vhat=vRecon(sf,vp,kernel_x,kernel_y)

    du=uhat-u
    dv=vhat-v

    dvp_u=fftconvolve(du,kernel_x,mode='full')
    dvp_v=fftconvolve(dv,kernel_y,mode='full')

    dsf_u=fftconvolve(du,-kernel_y,mode='full')
    dsf_v=fftconvolve(dv,kernel_x,mode='full')

    dsf=dsf_u+dsf_v
    dvp=dvp_u+dvp_v

    re=numpy.vstack([dsf[None,:,:,],dvp[None,:,:]])
    re=re.reshape(params.shape)
    re=re+lam*params/u.size
    #re=re+lam*params

    return re

def uv_transform(xc,yc,psi,chi,previous_u,previous_v):
    sf = numpy.pad(psi,(1,0),'edge')
    vp = numpy.pad(chi,(1,0),'edge')

    params=numpy.vstack([sf[None,:,:], vp[None,:,:]])

    pad_shape = params.shape
    lam = .001

    kernel_x = numpy.array([[-.5,.5],[-.5,.5]])/dxc[0]
    kernel_y = numpy.array([[-.5,-.5],[.5,.5]])/dxc[0]

    opt = optimize.minimize(costFunc,params,args=(u,v,kernel_x,kernel_y,pad_shape,lam),method='Newton-CG',jac=jac)
    params=opt.x.reshape(pad_shape)
    sf=params[0]
    vp=params[1]
    return uRecon(sf,vp,kernel_x,kernel_y),vRecon(sf,vp,kernel_x,kernel_y)

def psi_transform(Xc,Yc,Dxc,Dyc,u,v):
    # compute the potential phi
    intx=integrate.cumtrapz(v,Xc,axis=1,initial=0)[0]
    inty=integrate.cumtrapz(u,Yc,axis=0,initial=0)
    psi1=-intx+inty
    intx=integrate.cumtrapz(v,Xc,axis=1,initial=0)
    inty=integrate.cumtrapz(u,Yc,axis=0,initial=0)[:,0][:,None]
    psi2=-intx+inty
    psi = 0.5*(psi1+psi2)
    # psi = integrate.cumtrapz(Yc[:,0],u,initial = 0)*(Dyc[:,:]) - integrate.cumtrapz(Xc[0,:],v,initial = 0)*Dxc[:,:]
    return psi
def chi_transform(Xc,Yc,Dxc,Dyc,u,v):
    intx=integrate.cumtrapz(u,Xc,axis=1,initial=0)[0]
    inty=integrate.cumtrapz(v,Yc,axis=0,initial=0)
    chi1=intx+inty

    intx=integrate.cumtrapz(u,Xc,axis=1,initial=0)
    inty=integrate.cumtrapz(v,Yc,axis=0,initial=0)[:,0][:,None]
    chi2=intx+inty

    chi=0.5*(chi1+chi2)
    return chi

xc = numpy.linspace(0.,1.,12)
yc = numpy.linspace(0.,1.,12)
dxc = numpy.diff(xc)
dyc = numpy.diff(yc)
Xc, Yc = numpy.meshgrid(xc,yc)
Dxc, Dyc = numpy.meshgrid(dxc,dyc)
u = 3*Yc[:,:]**2-3*Xc[:,:]**2+Yc[:,:]
v = 6*Xc[:,:]*Yc[:,:] + Xc[:,:]
psi = psi_transform(Xc,Yc,Dxc,Dyc,u,v)
chi = chi_transform(Xc,Yc,Dxc,Dyc,u,v)
uhat, vhat = uv_transform(xc,yc,psi,chi,u,v)
# plt.contourf(psi)
# plt.show()
# plt.quiver(Xc,Yc,uhat,vhat,scale=10)
# plt.show()
