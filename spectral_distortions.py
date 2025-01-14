import numpy as np
import os
import sys
sys.path.append("/moto/hill/users/as6131/software/SZpack/")
import SZpack as SZ
from scipy import interpolate

### See components for a better description of the signals.

TCMB = 2.7255 #Kelvin
hplanck = 6.626070150e-34  # MKS
kboltz = 1.380649e-23  # MKS
clight=299792458.0 #MKS
m_elec = 510.999
jy = 1.e26
dT_factor=2.0*(kboltz*TCMB)**3.0/(hplanck*clight)**2.0*jy

ndp = np.float64

this_dir=os.path.dirname(os.path.abspath(__file__))

def DeltaI_DeltaT(freqs, DeltaT_amp=1.2e-4*TCMB):
    X = hplanck*freqs/(kboltz*TCMB)
    return (DeltaT_amp/TCMB * X**4.0 * np.exp(X)/(np.exp(X) - 1.0)**2.0 * 2.0*(kboltz*TCMB)**3.0 / (hplanck*clight)**2.0 * jy).astype(ndp)

def DeltaI_DeltaT_Abitbol17(freqs, DeltaT_amp=1.2e-4):
    X = hplanck*freqs/(kboltz*TCMB)
    return (DeltaT_amp * X**4.0 * np.exp(X)/(np.exp(X) - 1.0)**2.0 * 2.0*(kboltz*TCMB)**3.0 / (hplanck*clight)**2.0 * jy).astype(ndp)

def DeltaI_mu(freqs, mu_amp=2.e-8):
    X = hplanck*freqs/(kboltz*TCMB)
    return (mu_amp * (X / 2.1923 - 1.0)/X * X**4.0 * np.exp(X)/(np.exp(X) - 1.0)**2.0 * 2.0*(kboltz*TCMB)**3.0 / (hplanck*clight)**2.0 * jy).astype(ndp)

def DeltaI_reltSZ_2param_yweight(freqs, y_tot=1.77e-6, kT_yweight=1.245):
    tau = y_tot/kT_yweight * m_elec
    X = hplanck*freqs/(kboltz*TCMB)
    Xtwid = X*np.cosh(0.5*X)/np.sinh(0.5*X)
    Stwid = X/np.sinh(0.5*X)
    Y0=Xtwid-4.0
    Y1=-10.0+23.5*Xtwid-8.4*Xtwid**2+0.7*Xtwid**3+Stwid**2*(-4.2+1.4*Xtwid)
    Y2=-7.5+127.875*Xtwid-173.6*Xtwid**2.0+65.8*Xtwid**3.0-8.8*Xtwid**4.0+0.3666667*Xtwid**5.0+Stwid**2.0*(-86.8+131.6*Xtwid-48.4*Xtwid**2.0+4.7666667*Xtwid**3.0)+Stwid**4.0*(-8.8+3.11666667*Xtwid)
    Y3=7.5+313.125*Xtwid-1419.6*Xtwid**2.0+1425.3*Xtwid**3.0-531.257142857*Xtwid**4.0+86.1357142857*Xtwid**5.0-6.09523809524*Xtwid**6.0+0.15238095238*Xtwid**7.0+Stwid**2.0*(-709.8+2850.6*Xtwid-2921.91428571*Xtwid**2.0+1119.76428571*Xtwid**3.0-173.714285714*Xtwid**4.0+9.14285714286*Xtwid**5.0)+Stwid**4.0*(-531.257142857+732.153571429*Xtwid-274.285714286*Xtwid**2.0+29.2571428571*Xtwid**3.0)+Stwid**6.0*(-25.9047619048+9.44761904762*Xtwid)
    gfuncrel_only=Y1*(kT_yweight/m_elec)+Y2*(kT_yweight/m_elec)**2.0+Y3*(kT_yweight/m_elec)**3.0
    return (X**4.0 * np.exp(X)/(np.exp(X) - 1.0)**2.0 * 2.0*(kboltz*TCMB)**3.0 / (hplanck*clight)**2.0 * (y_tot * Y0 + tau * (kT_yweight/m_elec) * gfuncrel_only) * jy).astype(ndp)

def DeltaI_y(freqs, y_tot=1.77e-6):
    X = hplanck*freqs/(kboltz*TCMB)
    return ((y_tot * (X / np.tanh(X/2.0) - 4.0) * X**4.0 * np.exp(X)/(np.exp(X) - 1.0)**2.0 * 2.0*(kboltz*TCMB)**3.0 / (hplanck*clight)**2.0) * jy).astype(ndp)

def blackbody(nu, DT=1.e-3):
    T = DT*TCMB + TCMB
    X = hplanck * nu / (kboltz * T)
    Xcmb = hplanck * nu / (kboltz * TCMB)
    bbT = 2.0 * hplanck * (nu * nu * nu) / (clight ** 2) * (1.0 / (np.exp(X) - 1.0))
    bbTcmb = 2.0 * hplanck * (nu * nu * nu) / (clight ** 2) * (1.0 / (np.exp(Xcmb) - 1.0))
    return ( (bbT - bbTcmb)*jy ).astype(ndp)

def recombination(freqs, scale=1.0):
    rdata = np.loadtxt(this_dir+'templates/recombination/total_spectrum_f.dat')
    fs = rdata[:,0] * 1e9
    recomb = rdata[:,1]
    template = interpolate.interp1d(np.log10(fs), np.log10(recomb), fill_value=np.log10(1e-30), bounds_error=False)
    return scale * 10.0**template(np.log10(freqs))

###################################### additions ######################################
def DeltaI_reltSZ_w1(freqs, y_tot, kT_yweight, omega):
    Yorder=3
    #based on Abitbol+2017, Hill+2015, uses Y functions of Nozawa+2006, Itoh+1998
    yIGM_plus_yreion=1.87e-7
    X=hplanck*freqs/(kboltz*TCMB)
    Xtwid=X*np.cosh(0.5*X)/np.sinh(0.5*X)
    Stwid=X/np.sinh(0.5*X)

    #Y functions
    Y0=Xtwid-4.0
    Y1=-10.0+23.5*Xtwid-8.4*Xtwid**2+0.7*Xtwid**3+Stwid**2*(-4.2+1.4*Xtwid)
    Y2=-7.5+127.875*Xtwid-173.6*Xtwid**2.0+65.8*Xtwid**3.0-8.8*Xtwid**4.0+11.0/30.0*Xtwid**5.0\
    +Stwid**2.0*(-86.8+131.6*Xtwid-48.4*Xtwid**2.0+143.0/30.0*Xtwid**3.0)\
    +Stwid**4.0*(-8.8+187.0/60.0*Xtwid)

    Y3=7.5+313.125*Xtwid-1419.6*Xtwid**2.0+1425.3*Xtwid**3.0-18594.0/35.0*Xtwid**4.0+12059.0/140.0*Xtwid**5.0-128.0/21.0*Xtwid**6.0+16.0/105.0*Xtwid**7.0\
    +Stwid**2.0*(-709.8+2850.6*Xtwid-102267.0/35.0*Xtwid**2.0+156767.0/140.0*Xtwid**3.0-1216.0/7.0*Xtwid**4.0+64.0/7.0*Xtwid**5.0)\
    +Stwid**4.0*(-18594.0/35.0+205003.0/280.0*Xtwid-1920.0/7.0*Xtwid**2.0+1024.0/35.0*Xtwid**3.0)\
    +Stwid**6.0*(-544.0/21.0+992.0/105.0*Xtwid)
    #gfuncrel=Y0+Y1*(kT_yweight/const.m_elec)+Y2*(kT_yweight/const.m_elec)**2.0+Y3*(kT_yweight/const.m_elec)**3.0+Y4*(kT_yweight/const.m_elec)**4.0
    #add different y orders
    orders=np.array([Y0,Y1,Y2,Y3])
    gfuncrel=0.0
    for i in range(Yorder+1):
        gfuncrel+=orders[i]*(kT_yweight/m_elec)**i
        # print(f"added {i}")
    if Yorder==0:
        gfuncrel=Y0

    Trelapprox = (yIGM_plus_yreion * Y0 + (y_tot-yIGM_plus_yreion) * (gfuncrel+(Y2*(kT_yweight/m_elec)**2.0+3*Y3*(kT_yweight/m_elec)**3.0)*omega)) * (TCMB*1e6)
    Planckian = X**4.0*np.exp(X)/(np.exp(X) - 1.0)**2.0
    DeltaIrelapprox = Planckian*Trelapprox / (TCMB*1e6)

    return DeltaIrelapprox

def DeltaI_reltSZ(freqs, y_tot=1.77e-6, kT_yweight=1.245):
    Yorder=4
    #based on Abitbol+2017, Hill+2015, uses Y functions of Nozawa+2006, Itoh+1998
    yIGM_plus_yreion=1.87e-7
    X=hplanck*freqs/(kboltz*TCMB)
    Xtwid=X*np.cosh(0.5*X)/np.sinh(0.5*X)
    Stwid=X/np.sinh(0.5*X)

    #Y functions
    Y0=Xtwid-4.0
    Y1=-10.0+23.5*Xtwid-8.4*Xtwid**2+0.7*Xtwid**3+Stwid**2*(-4.2+1.4*Xtwid)
    Y2=-7.5+127.875*Xtwid-173.6*Xtwid**2.0+65.8*Xtwid**3.0-8.8*Xtwid**4.0+11.0/30.0*Xtwid**5.0\
    +Stwid**2.0*(-86.8+131.6*Xtwid-48.4*Xtwid**2.0+143.0/30.0*Xtwid**3.0)\
    +Stwid**4.0*(-8.8+187.0/60.0*Xtwid)

    Y3=7.5+313.125*Xtwid-1419.6*Xtwid**2.0+1425.3*Xtwid**3.0-18594.0/35.0*Xtwid**4.0+12059.0/140.0*Xtwid**5.0-128.0/21.0*Xtwid**6.0+16.0/105.0*Xtwid**7.0\
    +Stwid**2.0*(-709.8+2850.6*Xtwid-102267.0/35.0*Xtwid**2.0+156767.0/140.0*Xtwid**3.0-1216.0/7.0*Xtwid**4.0+64.0/7.0*Xtwid**5.0)\
    +Stwid**4.0*(-18594.0/35.0+205003.0/280.0*Xtwid-1920.0/7.0*Xtwid**2.0+1024.0/35.0*Xtwid**3.0)\
    +Stwid**6.0*(-544.0/21.0+992.0/105.0*Xtwid)

    Stwid2_term_Y4=Stwid**2.0*(-62391.0/20.0+614727.0/20.0*Xtwid-1368279.0/20.0*Xtwid**2.0+4624139.0/80.0*Xtwid**3.0-157396.0/7.0*Xtwid**4.0+30064.0/7.0*Xtwid**5.0-2717.0/7.0*Xtwid**6.0+2761.0/210.0*Xtwid**7.0)
    Stwid4_term_Y4=Stwid**4.0*(-124389.0/10.0+6046951.0/160.0*Xtwid-248520.0/7.0*Xtwid**2.0+481024.0/35.0*Xtwid**3.0-15972.0/7.0*Xtwid**4.0+18689.0/140.0*Xtwid**5.0)
    Stwid6_term_Y4=Stwid**6.0*(-70414.0/21.0+465992.0/105.0*Xtwid-11792.0/7.0*Xtwid**2.0+19778.0/105.0*Xtwid**3.0)
    Stwid8_term_Y4=Stwid**8.0*(-682.0/7.0+7601.0/210.0*Xtwid)

    Y4=-135.0/32.0+30375.0/128.0*Xtwid-62391.0/10.0*Xtwid**2.0+614727.0/40.0*Xtwid**3.0-12438.9*Xtwid**4.0+355703.0/80.0*Xtwid**5.0\
    -16568.0/21.0*Xtwid**6.0+7516.0/105.0*Xtwid**7.0-22.0/7.0*Xtwid**8.0+11.0/210.0*Xtwid**9.0\
    +Stwid2_term_Y4+Stwid4_term_Y4+Stwid6_term_Y4+Stwid8_term_Y4

    #gfuncrel=Y0+Y1*(kT_yweight/const.m_elec)+Y2*(kT_yweight/const.m_elec)**2.0+Y3*(kT_yweight/const.m_elec)**3.0+Y4*(kT_yweight/const.m_elec)**4.0
    #add different y orders
    orders=np.array([Y0,Y1,Y2,Y3,Y4])
    gfuncrel=0.0
    for i in range(Yorder+1):
        gfuncrel+=orders[i]*(kT_yweight/m_elec)**i
        # print(f"added {i}")
    if Yorder==0:
        gfuncrel=Y0

    Trelapprox = (yIGM_plus_yreion * Y0 + (y_tot-yIGM_plus_yreion) * gfuncrel) * (TCMB*1e6)
    Planckian = X**4.0*np.exp(X)/(np.exp(X) - 1.0)**2.0
    DeltaIrelapprox = Planckian*Trelapprox / (TCMB*1e6)

    return DeltaIrelapprox*dT_factor

def DeltaI_reltSZ_Y3(freqs, y_tot, kT_yweight):
    Yorder=3
    #based on Abitbol+2017, Hill+2015, uses Y functions of Nozawa+2006, Itoh+1998
    yIGM_plus_yreion=1.87e-7
    X=hplanck*freqs/(kboltz*TCMB)
    Xtwid=X*np.cosh(0.5*X)/np.sinh(0.5*X)
    Stwid=X/np.sinh(0.5*X)

    #Y functions
    Y0=Xtwid-4.0
    Y1=-10.0+23.5*Xtwid-8.4*Xtwid**2+0.7*Xtwid**3+Stwid**2*(-4.2+1.4*Xtwid)
    Y2=-7.5+127.875*Xtwid-173.6*Xtwid**2.0+65.8*Xtwid**3.0-8.8*Xtwid**4.0+11.0/30.0*Xtwid**5.0\
    +Stwid**2.0*(-86.8+131.6*Xtwid-48.4*Xtwid**2.0+143.0/30.0*Xtwid**3.0)\
    +Stwid**4.0*(-8.8+187.0/60.0*Xtwid)

    Y3=7.5+313.125*Xtwid-1419.6*Xtwid**2.0+1425.3*Xtwid**3.0-18594.0/35.0*Xtwid**4.0+12059.0/140.0*Xtwid**5.0-128.0/21.0*Xtwid**6.0+16.0/105.0*Xtwid**7.0\
    +Stwid**2.0*(-709.8+2850.6*Xtwid-102267.0/35.0*Xtwid**2.0+156767.0/140.0*Xtwid**3.0-1216.0/7.0*Xtwid**4.0+64.0/7.0*Xtwid**5.0)\
    +Stwid**4.0*(-18594.0/35.0+205003.0/280.0*Xtwid-1920.0/7.0*Xtwid**2.0+1024.0/35.0*Xtwid**3.0)\
    +Stwid**6.0*(-544.0/21.0+992.0/105.0*Xtwid)

    #gfuncrel=Y0+Y1*(kT_yweight/const.m_elec)+Y2*(kT_yweight/const.m_elec)**2.0+Y3*(kT_yweight/const.m_elec)**3.0+Y4*(kT_yweight/const.m_elec)**4.0
    #add different y orders
    orders=np.array([Y0,Y1,Y2,Y3])
    gfuncrel=0.0
    for i in range(Yorder+1):
        gfuncrel+=orders[i]*(kT_yweight/m_elec)**i
        # print(f"added {i}")
    if Yorder==0:
        gfuncrel=Y0

    Trelapprox = (yIGM_plus_yreion * Y0 + (y_tot-yIGM_plus_yreion) * gfuncrel) * (TCMB*1e6)
    Planckian = X**4.0*np.exp(X)/(np.exp(X) - 1.0)**2.0
    DeltaIrelapprox = Planckian*Trelapprox / (TCMB*1e6)

    return DeltaIrelapprox*dT_factor

def DeltaI_reltSZ_Y2(freqs, y_tot, kT_yweight):
    Yorder=2
    #based on Abitbol+2017, Hill+2015, uses Y functions of Nozawa+2006, Itoh+1998
    yIGM_plus_yreion=1.87e-7
    X=hplanck*freqs/(kboltz*TCMB)
    Xtwid=X*np.cosh(0.5*X)/np.sinh(0.5*X)
    Stwid=X/np.sinh(0.5*X)

    #Y functions
    Y0=Xtwid-4.0
    Y1=-10.0+23.5*Xtwid-8.4*Xtwid**2+0.7*Xtwid**3+Stwid**2*(-4.2+1.4*Xtwid)
    Y2=-7.5+127.875*Xtwid-173.6*Xtwid**2.0+65.8*Xtwid**3.0-8.8*Xtwid**4.0+11.0/30.0*Xtwid**5.0\
    +Stwid**2.0*(-86.8+131.6*Xtwid-48.4*Xtwid**2.0+143.0/30.0*Xtwid**3.0)\
    +Stwid**4.0*(-8.8+187.0/60.0*Xtwid)
    #gfuncrel=Y0+Y1*(kT_yweight/const.m_elec)+Y2*(kT_yweight/const.m_elec)**2.0+Y3*(kT_yweight/const.m_elec)**3.0+Y4*(kT_yweight/const.m_elec)**4.0
    #add different y orders
    orders=np.array([Y0,Y1,Y2])
    gfuncrel=0.0
    for i in range(Yorder+1):
        gfuncrel+=orders[i]*(kT_yweight/m_elec)**i
        # print(f"added {i}")
    if Yorder==0:
        gfuncrel=Y0

    Trelapprox = (yIGM_plus_yreion * Y0 + (y_tot-yIGM_plus_yreion) * gfuncrel) * (TCMB*1e6)
    Planckian = X**4.0*np.exp(X)/(np.exp(X) - 1.0)**2.0
    DeltaIrelapprox = Planckian*Trelapprox / (TCMB*1e6)

    return DeltaIrelapprox*dT_factor

def DeltaI_reltSZ_Y1(freqs, y_tot, kT_yweight):
    Yorder=1
    #based on Abitbol+2017, Hill+2015, uses Y functions of Nozawa+2006, Itoh+1998
    yIGM_plus_yreion=1.87e-7
    X=hplanck*freqs/(kboltz*TCMB)
    Xtwid=X*np.cosh(0.5*X)/np.sinh(0.5*X)
    Stwid=X/np.sinh(0.5*X)

    #Y functions
    Y0=Xtwid-4.0
    Y1=-10.0+23.5*Xtwid-8.4*Xtwid**2+0.7*Xtwid**3+Stwid**2*(-4.2+1.4*Xtwid)
    #gfuncrel=Y0+Y1*(kT_yweight/const.m_elec)+Y2*(kT_yweight/const.m_elec)**2.0+Y3*(kT_yweight/const.m_elec)**3.0+Y4*(kT_yweight/const.m_elec)**4.0
    #add different y orders
    orders=np.array([Y0,Y1])
    gfuncrel=0.0
    for i in range(Yorder+1):
        gfuncrel+=orders[i]*(kT_yweight/m_elec)**i
        # print(f"added {i}")
    if Yorder==0:
        gfuncrel=Y0

    Trelapprox = (yIGM_plus_yreion * Y0 + (y_tot-yIGM_plus_yreion) * gfuncrel) * (TCMB*1e6)
    Planckian = X**4.0*np.exp(X)/(np.exp(X) - 1.0)**2.0
    DeltaIrelapprox = Planckian*Trelapprox / (TCMB*1e6)

    return DeltaIrelapprox*dT_factor

def nu_to_x(f):
    return hplanck*f/(kboltz*TCMB)

def DeltaI_rel_SZpack(freqs, y_tot, kT_yweight):

    x_array=nu_to_x(freqs)

    if kT_yweight<=70:
        dI_rel=SZ.compute_combo_from_variables(x_array, kT_yweight/m_elec, 0, 0, 10, 0, "monopole")
    else:
        #rel_corr_file=np.load(f"/moto/home/as6131/firas_distortions/data/3D_rel_precompute_70.0_1000.0_0pt1.npy", allow_pickle=True).item()
        rel_corr_file=np.load(f"/moto/home/as6131/firas_distortions/data/3D_rel_precompute_70_200_0pt1.npy", allow_pickle=True).item()
        interp_func=interpolate.interp1d(rel_corr_file['kTe'], rel_corr_file['dI'])
        dI_rel=interp_func(kT_yweight)

    theta=kT_yweight/m_elec
    tau=y_tot/theta
	
    return dI_rel*tau*10**6

def DeltaI_rel_SZpack_5D(freqs, y_tot, kT_yweight):
    
    x_array=nu_to_x(freqs)
    rel_5D=SZ.Integral5D.compute_from_variables(x_array, kT_yweight/m_elec, 0., 0., 1.0e-6, "monopole")
    theta=kT_yweight/m_elec
    tau=y_tot/theta

    return rel_5D*tau*10**6

def DeltaI_rel_SZpack_3D(freqs, y_tot, kT_yweight):

    x_array=nu_to_x(freqs)
    rel_3D=SZ.Integral3D.compute_from_variables(x_array, kT_yweight/m_elec, 0., 0., 1.0e-4, "monopole")
    theta=kT_yweight/m_elec
    tau=y_tot/theta

    return rel_3D*tau*10**6
###################################### additions ######################################
