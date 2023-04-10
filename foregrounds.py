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
print(this_dir)

def jens_synch_rad(nu, As=288., alps=-0.82, w2s=0.2):
    nu0s = 100.e9
    return (As * (nu / nu0s) ** alps * (1. + 0.5 * w2s * np.log(nu / nu0s) ** 2) * jy).astype(ndp)

def jens_synch_rad_fixed_curv(nu, As=288., alps=-0.82):
    w2s=0.2
    nu0s = 100.e9
    return (As * (nu / nu0s) ** alps * (1. + 0.5 * w2s * np.log(nu / nu0s) ** 2) * jy).astype(ndp)

def jens_synch_rad_no_curv(nu, As=288., alps=-0.82):
    nu0s = 100.e9
    return (As * (nu / nu0s) ** alps * jy).astype(ndp)

def jens_synch_rad_no_curv_fixed_index_m3pt1(nu, As=288.):
    alps=-3.1
    nu0s = 100.e9
    return (As * (nu / nu0s) ** alps * jy).astype(ndp)

def jens_synch_rad_fixed_curv_and_index(nu, As=288.):
    alps=-0.82
    w2s=0.2
    nu0s = 100.e9
    return (As * (nu / nu0s) ** alps * (1. + 0.5 * w2s * np.log(nu / nu0s) ** 2) * jy).astype(ndp)

def jens_synch_rad_no_curv_and_fixed_index_m3pt1(nu, As=288.):
    alps=-3.1
    nu0s = 100.e9
    return (As * (nu / nu0s) ** alps * jy).astype(ndp)

def jens_synch_rad1(nu, As=288., alps=-0.82):
    nu0s = 100.e9
    return (As * (nu / nu0s) ** alps * jy).astype(ndp)

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

def thermal_dust_rad(nu, Ad=1.36e6, Bd=1.53, Td=21.):
    X = hplanck * nu / (kboltz * Td)
    return (Ad * X**Bd * X**3. / (np.exp(X) - 1.0) * jy).astype(ndp)

def thermal_dust_rad_fixed_beta_1pt6_Td_19pt6(nu, Ad=1.36e6):
    Td=19.6
    Bd=1.6
    X = hplanck * nu / (kboltz * Td)
    return (Ad * X**Bd * X**3. / (np.exp(X) - 1.0) * jy).astype(ndp)

def thermal_dust_rad_fixed_beta(nu, Ad=1.36e6,Td=21.):
    Bd=1.53
    X = hplanck * nu / (kboltz * Td)
    return (Ad * X**Bd * X**3. / (np.exp(X) - 1.0) * jy).astype(ndp)

def thermal_dust_rad_fixed_Td(nu, Ad=1.36e6, Bd=1.53):
    Td=21.
    X = hplanck * nu / (kboltz * Td)
    return (Ad * X**Bd * X**3. / (np.exp(X) - 1.0) * jy).astype(ndp)

def thermal_dust_rad_fixed_Td_19pt6(nu, Ad=1.36e6, Bd=1.53):
    Td=19.6
    X = hplanck * nu / (kboltz * Td)
    return (Ad * X**Bd * X**3. / (np.exp(X) - 1.0) * jy).astype(ndp)

def cib_rad(nu, Acib=3.46e5, Bcib=0.86, Tcib=18.8):
    X = hplanck * nu / (kboltz * Tcib)
    return (Acib * X**Bcib * X**3. / (np.exp(X) - 1.0) * jy).astype(ndp)

def cib_rad_fixed_beta(nu, Acib=3.46e5,Tcib=18.8):
    Bcib=0.86,
    X = hplanck * nu / (kboltz * Tcib)
    return (Acib * X**Bcib * X**3. / (np.exp(X) - 1.0) * jy).astype(ndp)

def cib_rad_fixed_Td(nu, Acib=3.46e5, Bcib=0.86):
    Tcib=18.8
    X = hplanck * nu / (kboltz * Tcib)
    return (Acib * X**Bcib * X**3. / (np.exp(X) - 1.0) * jy).astype(ndp)

def cib_rad_fixed_beta_and_Td(nu, Acib=3.46e5):
    Bcib=0.86
    Tcib=18.8
    X = hplanck * nu / (kboltz * Tcib)
    return (Acib * X**Bcib * X**3. / (np.exp(X) - 1.0) * jy).astype(ndp)

def co_rad(nu, Aco=1.):
    x = np.load(this_dir+'/templates/co_arrays.npy').astype(ndp)
    freqs = x[0]
    co = x[1]
    fs = interpolate.interp1d(log10(freqs), log10(co), bounds_error=False, fill_value="extrapolate")
    return (Aco * 10. ** fs(log10(nu)) * jy).astype(ndp)

def dust_moments_omega22(nu, Adm=1.36e6, alphadm=1.53, Tdm=21., omega22=0.1):
    X = hplanck * nu / (kboltz * Tdm)
    nu0 = (kboltz * Tdm)/hplanck
    lnnu = np.log(nu/nu0)
    Y1 = X * np.exp(X) / (np.exp(X) - 1.)
    Y2 = Y1*X*np.cosh(X/2.)/np.sinh(X/2.)
    zeroth = Adm * X**alphadm * X**3 / (np.exp(X) - 1.)
    return zeroth * (1. + 0.5 * omega22 * lnnu**2)

def dust_moments_omega22_omega23(nu, Adm=1.36e6, alphadm=1.53, Tdm=21., omega22=0.1, omega23=0.1):
    X = hplanck * nu / (kboltz * Tdm)
    nu0 = (kboltz * Tdm)/hplanck
    lnnu = np.log(nu/nu0)
    Y1 = X * np.exp(X) / (np.exp(X) - 1.)
    zeroth = Adm * X**alphadm * X**3 / (np.exp(X) - 1.)
    return zeroth * (1. + 0.5 * omega22 * lnnu**2+omega23*lnnu*Y1)

def dust_moments_omega22_omega23_omega33(nu, Adm=1.36e6, alphadm=1.53, Tdm=21., omega22=0.1, omega23=0.1,omega33=0.1):
    X = hplanck * nu / (kboltz * Tdm)
    nu0 = (kboltz * Tdm)/hplanck
    lnnu = np.log(nu/nu0)
    Y1 = X * np.exp(X) / (np.exp(X) - 1.)
    Y2 = Y1*X*np.cosh(X/2.)/np.sinh(X/2.)
    zeroth = Adm * X**alphadm * X**3 / (np.exp(X) - 1.)
    return zeroth * (1. + 0.5 * omega22 * lnnu**2+omega23*lnnu*Y1+0.5*omega33*Y2)

def dust_moments_2nd_order(nu, Adm=1.36e6, alphadm=1.53, Tdm=21.,omega22=0.1, omega23=0.1,omega33=0.1,omega222=0.1, omega223=0.1, omega233=0.1, omega333=0.1):
    X = hplanck * nu / (kboltz * Tdm)
    nu0 = kboltz *Tdm/hplanck
    lnnu = np.log(nu/nu0)
    Y1 = X * np.exp(X) / (np.exp(X) - 1.)
    Y2 = Y1*X*np.cosh(X/2.)/np.sinh(X/2.)
    Y3 = Y2*X*(np.cosh(X)+5.)/(np.cosh(X)-1)
    zeroth = Adm * X**alphadm * X**3 / (np.exp(X) - 1.)
    return zeroth * (1. + 0.5 * omega22 * lnnu**2+omega23*lnnu*Y1+0.5*omega33*Y2+1/6.*omega222*lnnu**3+0.5*omega223*lnnu**2*Y1+0.5*omega233*lnnu*Y2+1/6.*omega333*Y3)

def dust_moments_omega22_omega23_omega33_allzero(nu, Adm=1.36e6, alphadm=1.53, Tdm=21.):
    omega22=0
    omega23=0
    omega33=0
    X = hplanck * nu / (kboltz * Tdm)
    nu0 = (kboltz * Tdm)/hplanck
    lnnu = np.log(nu/nu0)
    Y1 = X * np.exp(X) / (np.exp(X) - 1.)
    Y2 = Y1*X*np.cosh(X/2.)/np.sinh(X/2.)
    zeroth = Adm * X**alphadm * X**3 / (np.exp(X) - 1.)
    return zeroth * (1. + 0.5 * omega22 * lnnu**2+omega23*lnnu*Y1+0.5*omega33*Y2)

def dust_moments(nu, Adm=3.2e-4, alphadm=1.22, Tdm=21.1, omega1=0.09):
    X = hplanck * nu / (kboltz * Tdm)
    nu0 = 100.e9
    lnnu = np.log(nu/nu0)
    Y1 = X * np.exp(X) / (np.exp(X) - 1.)
    zeroth = Adm * (nu/nu0)**alphadm * nu**3 / (np.exp(X) - 1.)
    return zeroth * (1. + 0.5 * omega1 * lnnu**2) * 1.e-26
