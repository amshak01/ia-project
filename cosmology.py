import numpy as np
#from galsim.integ import int1d
from scipy.integrate import quad as scipy_int1d
import warnings
tol=1.e-9
class cosmology():
    def __init__(self,h=1,Omega_b=0.05,Omega_dm=0.25, Omega_L=0.7,Omega_R=1.e-5,Omega_k=0,
                dz=0.001,z_max=2,do_calcs=1,w0=-1,wa=0,rtol=1.e-4,h_inv=True,**kwargs):
        self.h=h
        self.rtol=rtol #tolerence for scipy.integrate.quadrature
        self.w0=w0
        self.wa=wa
        self.Omega_b=Omega_b
        self.Omega_dm=Omega_dm-Omega_R #negligible change
        self.Omega_L=Omega_L
        self.Omega_k=Omega_k
        self.Omega_R=Omega_R
        self.h_inv=h_inv
        self.dz=0.001
        self.c=3*10**5
        self.Omega_m=self.Omega_b+self.Omega_dm
        if not np.isclose(self.Omega_m+self.Omega_L+self.Omega_R+self.Omega_k,1.0):
            #print self.Omega_m+self.Omega_L+self.Omega_R
            raise Exception('Sum Omega='+str(self.Omega_m+self.Omega_L+self.Omega_R+self.Omega_k)+'!=1')
        self.z_max=z_max
        self.z=np.arange(start=0,stop=z_max,step=self.dz)
        self.H0=self.h*100.0
        self.Dh=self.c/self.H0
        if self.h_inv:
            self.Dh=self.c/100.
        self.rho=self.Rho_crit()*self.Omega_m
        if self.Omega_k!=0:
            self.Dh=self.c*(np.sqrt(np.absolute(self.Omega_k))) #a0h0=1./sqrt(Omega_k)
        if do_calcs!=0:
            self.Ez=self.E_z(self.z)
            self.Hz=self.H0*self.Ez
            self.Dc=self.DC(self.z)
            self.Dm=self.DM(self.z)
            self.Da=self.DA(self.z)
            self.Dz=self.DZ_int(self.z)
            self.fz=-1.*(1+self.z)/(self.Dz)*np.gradient(self.Dz,self.dz)
            self.Omega_m_z=self.Omega_Z(self.z)

    def E_z(self,z):
        z=np.array(z)
        if self.wa!=0:
            return self.E_z_wa(z)
        return np.sqrt(self.Omega_m*(1+z)**3+self.Omega_k*(1+z)**2+self.Omega_R*(1+z)**4+self.Omega_L*(1+z)**(3*(1+self.w0)))

    def E_z_inv(self,z,Ez_func=None): #1/E_z
        if not Ez_func:
            Ez_func=self.E_z
        return 1./Ez_func(z)

    def H_z(self,z,Ez_func=None):#hubble parameter for redshift z
        if not Ez_func:
            Ez_func=self.E_z
        return self.H0*Ez_func(z)

    def w_z(self,z):
        return self.w0+self.wa*z/(1+z) # w=w0+wa*(1-a)

    def E_z_wa(self,z):
        def DE_int_funct(z2):
            return (1+self.w_z(z2))/(1+z2) #huterer & turner 2001
        if hasattr(z, "__len__"):
            j=0
            ez=np.zeros_like(z,dtype='float64')
            for i in z:
                try:
                    DE_int=int1d(DE_int_funct,0,i)
                except:
                    DE_int=scipy_int1d(DE_int_funct,0,i,epsrel=self.rtol,epsabs=tol)[0]
                ez[j]= np.sqrt(self.Omega_m*(1+i)**3+self.Omega_k*(1+i)**2+self.Omega_R*(1+i)**4 + self.Omega_L*np.exp(DE_int))
                #print i,DE_int,self.Omega_m*(1+i)**3+self.Omega_k*(1+i)**2+self.Omega_R*(1+i)**4,ez[j]
                j+=1
        else:
            try:
                DE_int=int1d(DE_int_funct,0,z)
            except:
                DE_int=scipy_int1d(DE_int_funct,0,z,epsrel=self.rtol,epsabs=tol)[0]

            ez= np.sqrt(self.Omega_m*(1+z)**3+ self.Omega_k*(1+z)**2+ self.Omega_R*(1+z)**4
                        +self.Omega_L*DE_int)
        return ez

    def DC(self,z=[0]): #line of sight comoving distance
        #z=np.array(z)
        #if len(z)>1:
        if hasattr(z, "__len__"):
            j=0
            ez=np.zeros_like(z)
            for i in z:

                try:
                    ezi=int1d(self.E_z_inv,0,i)
                except:
                    ezi=scipy_int1d(self.E_z_inv,0,i,epsrel=self.rtol,epsabs=tol)[0]#integration for array (vector) of z
                if len(z)==1:
                    ez=ezi
                else:
                    ez[j]=ezi
                #print j,i,ezi,ez[j]
                j=j+1
        else:
            try:
                ez=int1d(self.E_z_inv,0,z)
            except:
                ez=scipy_int1d(self.E_z_inv,0,z,epsrel=self.rtol,epsabs=tol)[0]#integration for scalar z
        return ez*self.Dh

    def DM(self,z=[0]): #transverse comoving distance
        Dc=self.DC(z)
        if self.Omega_k==0:
            return Dc
        curvature_radius=self.Dh/np.sqrt(np.absolute(self.Omega_k))
        if self.Omega_k>0:
            return curvature_radius*np.sinh(Dc/curvature_radius)
        if self.Omega_k<0:
            return curvature_radius*np.sin(Dc/curvature_radius)

    def DM2(self,z1=[0],z2=[]): #transverse comoving distance between 2 redshift
        Dc1=self.DC(z1)
        Dc2=self.DC(z2)
        Dc=Dc2-Dc1
        if self.Omega_k==0:
            return Dc
        curvature_radius=self.Dh/np.sqrt(np.absolute(self.Omega_k))
        if self.Omega_k>0:
            return curvature_radius*np.sinh(Dc/curvature_radius)
        if self.Omega_k<0:
            #warnings.warn('This formula for DM12 is apparently not valid for Omega_k<0')#http://arxiv.org/pdf/astro-ph/9603028v3.pdf... this function doesnot work if E(z)=0 between z1,z2
            return curvature_radius*np.sin(Dc/curvature_radius)

    def DA(self,z=[0]):#angular diameter distance
        Dm=self.DM(z)
        return Dm/(1+z)

    def DA2(self,z1=[],z2=[]):
        return self.DM2(z1=z1,z2=z2)/(1+z2)

    def DZ_approx(self,z=[0]):# linear growth factor.. only valid for LCDM
#fitting formula (eq 67) given in lahav and suto:living reviews in relativity.. http://www.livingreviews.org/lrr-2004-8
        hr=self.E_z_inv(z)
        omega_z=self.Omega_m*((1+z)**3)*((hr)**2)
        lamda_z=self.Omega_L*(hr**2)
        gz=5.0*omega_z/(2.0*(omega_z**(4.0/7.0)-lamda_z+(1+omega_z/2.0)*(1+lamda_z/70.0)))
        dz=gz/(1.0+z)
        return dz/dz[0]   #check normalisation

    def DZ_int(self,z=[0],Ez_func=None): #linear growth factor.. full integral.. eq 63 in Lahav and suto
        def intf(z):
            return (1+z)/self.H_z(z=z,Ez_func=Ez_func)**3
        j=0
        dz=np.zeros_like(z)
        inf=np.inf
        #inf=1.e10
        for i in z:
            try:
                dz[j]=self.H_z(i)*int1d(intf,i,inf)
            except:
                dz[j]=self.H_z(i)*scipy_int1d(intf,i,inf,epsrel=self.rtol,epsabs=tol)[0]
            j=j+1
        dz=dz*2.5*self.Omega_m*self.H0**2
        return dz/dz[0] #check for normalization

    def Omega_Z(self,z=[0]):
        z=np.array(z)
        omz=(self.H0**2/self.H_z(z)**2)*self.Omega_m*(1+z)**3 #omega_z=rho(z)/rho_crit(z)
        return omz

    def f_z(self,z=[0]):
        return self.Omega_Z(z=z)**0.55 #fitting func

    def f_z0(self,z=[0]): #from full func f(z)=a/D(dD/da), =-(1+z)dD/dz.. full integral here but slow.. better to interpolate self.fz
        fz=np.zeros_like(z)
        j=0
        for i in z:
            z2=np.linspace(i-0.01,i+0.01,101)
            Dz=self.DZ_int(z=z2)
            f2=-1.*(1+z2)/(Dz)*np.gradient(Dz,z2[1]-z2[0])
            fz[j]=f2[50]
            j+=1
        return fz

    def EG_z(self,z=[0]):
        return self.Omega_m/self.f_z(z=z)

    def Rho_crit(self):
        H0=self.H0*10**-6  #km/s/mpc ->km/s/pc
        if self.h_inv:
            H0=100.*10**-6  #km/s/mpc ->km/s/pc
        G=4.302*10**-3 #pc Msun km/s
        rc=3*H0**2/(8*np.pi*G)
        rc=rc*10**6 # unit of Msun/pc^2/mpc ... upsilon wgg in mpc
        return rc

    def comoving_volume(self,z=[]): #z should be bin edges
        z_mean=0.5*(z[1:]+z[:-1])
        dc=self.DC(z)
        dc_mean=self.DC(z_mean)
        return (dc_mean**2)*(dc[1:]-dc[:-1])

    def sigma_crit(self,zl=[],zs=[]):
        ds=self.DC(z=zs)
        dl=self.DC(z=zl)
        ddls=1-np.multiply.outer(1./ds,dl)#(ds-dl)/ds
        wt=(3./2.)*((self.H0/self.c)**2)*(1+zl)*dl/self.Rho_crit()
        wt=1./(ddls*wt)
        x=ddls<=0 #zs<zl
        wt[x]=np.inf
        return wt

    def sigma_crit_inv(self,zl=[],zs=[]):
        sig_c=self.sigma_crit(zl=zl,zs=zs)
        inv=1./sig_c
        return inv

    def lensing_efficiency(self,zl=[],zs=[]):
        sig_crit=self.sigma_crit(zl=zl,zs=zs)
        #H_z=self.H_z(z=zl)
        W_l=self.rho/sig_crit #*self.c/H_z
#        x=sig_crit<=0
 #       W_l[x]=0
        return W_l
#cc=cosmology(dz=0.1,h=1,z_max=15)
