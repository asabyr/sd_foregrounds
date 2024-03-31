import numpy as np
from numpy import log10
from scipy import interpolate
import os

hplanck = 6.626070150e-34  # MKS
kboltz = 1.380649e-23  # MKS
#jy = 1.e-26
jy = 1.
ndp = np.float64
this_dir=os.path.dirname(os.path.abspath(__file__))

###### new amplitudes because nu0 is fixed #####
Ad_x=1.36e6
Ad_353=Ad_x*(353.0e9*hplanck/(kboltz*21.0))**(1.53+3)

Acib_x=3.46e5
Acib_353=Acib_x*(353.0e9*hplanck/(kboltz*18.8))**(0.86+3)
###### new amplitudes because nu0 is fixed #####

def jens_synch_rad(nu, As=288., alps=-0.82, w2s=0.2):
    nu0s = 100.e9
    return (As * (nu / nu0s) ** alps * (1. + 0.5 * w2s * np.log(nu / nu0s) ** 2) * jy).astype(ndp)

def jens_freefree_rad(nu, EM=300.):
    Te = 7000.
    Teff = (Te / 1.e3) ** (3. / 2)
    nuff = 255.33e9 * Teff
    gff = 1. + np.log(1. + (nuff / nu) ** (np.sqrt(3) / np.pi))
    return (EM * gff * jy).astype(ndp)

def spinning_dust(nu, Asd=1.):
    ame_file = np.load(this_dir+'/templates/ame.npy').astype(ndp)
    ame_nu = ame_file[0]
    ame_I = ame_file[1]
    fsd = interpolate.interp1d(log10(ame_nu), log10(ame_I), bounds_error=False, fill_value="extrapolate")
    return (Asd * 10.**fsd(log10(nu)) * 1.e26).astype(ndp)

def thermal_dust_rad_Abitbol17(nu, Ad=1.36e6, Bd=1.53, Td=21.):
    X = hplanck * nu / (kboltz * Td)
    return (Ad * X**Bd * X**3. / (np.exp(X) - 1.0) * jy).astype(ndp)

def cib_rad_Abitbol17(nu, Acib=3.46e5, Bcib=0.86, Tcib=18.8):
    X = hplanck * nu / (kboltz * Tcib)
    return (Acib * X**Bcib * X**3. / (np.exp(X) - 1.0) * jy).astype(ndp)

def co_rad(nu, Aco=1.):
    x = np.load(this_dir+'/templates/co_arrays.npy').astype(ndp)
    freqs = x[0]
    co = x[1]
    fs = interpolate.interp1d(log10(freqs), log10(co), bounds_error=False, fill_value="extrapolate")
    return (Aco * 10. ** fs(log10(nu)) * jy).astype(ndp)

###################################### additions ######################################
def thermal_dust_rad(nu, Ad=Ad_353, Bd=1.53, Td=21.):
    X = hplanck * nu / (kboltz * Td)
    nu0=353.0*10.0**9
    return (Ad * (nu/nu0)**(Bd+3.0) / (np.exp(X) - 1.0) * jy).astype(ndp)

def jens_synch_rad_no_curv(nu, As=288., alps=-0.82):
    nu0s = 100.e9
    return (As * (nu / nu0s) ** alps * jy).astype(ndp)

def powerlaw(nu, As=300., alps=1):
    nu0s = 100.e9
    return As * (nu / nu0s) ** alps

def cib_rad(nu, Acib=Acib_353, Bcib=0.86, Tcib=18.8):
    X = hplanck * nu / (kboltz * Tcib)
    nu0=353.0*10.0**9
    return (Acib * (nu/nu0)**(Bcib+3.0) / (np.exp(X) - 1.0) * jy).astype(ndp)

def cib_rad_A17(nu, Acib=Acib_353):
    Bcib=0.86
    Tcib=18.8
    X = hplanck * nu / (kboltz * Tcib)
    nu0=353.0*10.0**9
    return (Acib * (nu/nu0)**(Bcib+3.0) / (np.exp(X) - 1.0) * jy).astype(ndp)

def cib_rad_MH23(nu, Acib=Acib_353):
    Bcib=1.59
    Tcib=11.95
    X = hplanck * nu / (kboltz * Tcib)
    nu0=353.0*10.0**9
    return (Acib * (nu/nu0)**(Bcib+3.0) / (np.exp(X) - 1.0) * jy).astype(ndp)

def dust_moments_omega2_omega3(nu, Ad=Ad_353, omega2=0.1, omega3=0.1):
    Td=21
    Bd=1.51
    X = hplanck * nu / (kboltz * Td)
    nu0 = 353.0e9
    dIdbeta = np.log(nu/nu0)
    dIdT = X * np.exp(X) / (np.exp(X) - 1.)/Td
    zeroth = Ad * X**Bd * X**3 / (np.exp(X) - 1.)

    return zeroth * (1.+omega2*dIdbeta+omega3*dIdT)

def dust_moments_omega2_omega3_bestfit(nu, Ad=Ad_353, omega2=0.1, omega3=0.1):
    Td=21
    Bd=1.51
    Td=9.86515065e-01
    Bd=2.37256467e+01
    X = hplanck * nu / (kboltz * Td)
    nu0 = 353.0e9
    dIdbeta = np.log(nu/nu0)
    dIdT = X * np.exp(X) / (np.exp(X) - 1.)/Td
    zeroth = Ad * X**Bd * X**3 / (np.exp(X) - 1.)

    return zeroth * (1.+omega2*dIdbeta+omega3*dIdT)

def dust_moments_omega2_omega3_omega22(nu, Ad=Ad_353, omega2=0.1, omega3=0.1, omega22=0.1):
    Td=21
    Bd=1.51
    X = hplanck * nu / (kboltz * Td)
    nu0 = 353.0e9
    dIdbeta = np.log(nu/nu0)
    dIdT = X * np.exp(X) / (np.exp(X) - 1.)/Td
    zeroth = Ad * X**Bd * X**3 / (np.exp(X) - 1.)

    return zeroth * (1.+omega2*dIdbeta+omega3*dIdT+omega22*dIdbeta**2.0)
###################################### additions ######################################

def dust_moments(nu, Adm=3.2e-4, alphadm=1.22, Tdm=21.1, omega1=0.09):
    X = hplanck * nu / (kboltz * Tdm)
    nu0 = 100.e9
    lnnu = np.log(nu/nu0)
    Y1 = X * np.exp(X) / (np.exp(X) - 1.)
    zeroth = Adm * (nu/nu0)**alphadm * nu**3 / (np.exp(X) - 1.)
    return zeroth * (1. + 0.5 * omega1 * lnnu**2) * 1.e-26
