import inspect
import numpy as np
from scipy import interpolate
import os
import spectral_distortions as sd
import foregrounds as fg
ndp = np.float64
this_dir=os.getcwd()
this_dir=os.path.dirname(os.path.abspath(__file__))
import sys
firas_code_dir=this_dir.replace('software/sd_foregrounds','firas_distortions/code/')
sys.path.append(firas_code_dir)
from read_data import remove_lines, prepare_data_lowf_masked_nolines, prepare_data_highf_masked_nolines
import copy

import matplotlib.pyplot as plt

N_pixels=3072.0
C_extra_factor=1.7

class FisherEstimation:
    def __init__(self, fmin=7.5e9, fmax=3.e12, fstep=15.e9, \
                 duration=86.4, bandpass=True, fsky=0.7, mult=1., \
                 priors={'alps':0.1, 'As':0.1}, drop=0, doCO=False, \
                #firas project additions
                instrument='firas', 
                fname='monopole_firas_freq_data_healpix_orthstipes_True_20230509.pkl', #monopole file, assumes it is in FIRAS code directory in /data
                file_type='monopole', # monopole or noise (uses monopole or covariance file)
                firas_method='invvar', # monopole method
                low_or_high='lowf', #which frequencies to use
                highf_thresh=1890, #if both or highf, then indicate upper bound
                lowf_mask=[2,-1], #which channels to throw out in lowf
                highf_mask=3, #which channels to throw out in highf
                which_noise='tot', #which part of covariance to use 'C','beta','JCJ', 'PEP','PUP','PTP'
                remove_lines=True, #remove channels near emission lines
                arg_dict={}): #sky model parameters 
                #!!!important!!
                # arg_dict has to be in the order that sky model functions are given to fisher
                # (or in the order of the default "fncs" array if no functions are specified)
                # !!!important!!

        self.fmin = fmin
        self.fmax = fmax
        self.bandpass_step = 1.e8
        self.fstep = fstep
        self.duration = duration
        self.bandpass = bandpass
        self.fsky = fsky #also FIRAS option
        self.mult = mult
        self.priors = priors
        self.drop = drop

        #firas project additions
        self.instrument=instrument
        self.fname=fname
        self.method=firas_method
        self.low_or_high=low_or_high
        self.highf_thresh=highf_thresh
        self.lowf_mask=lowf_mask
        self.highf_mask=highf_mask
        self.file_type=file_type
        self.which_noise=which_noise
        self.remove_lines=remove_lines
        self.arg_dict=arg_dict

        self.setup()
        self.set_signals()

        if instrument=='pixie' or 'pixie2024':
            if doCO:
                self.mask = ~np.isclose(115.27e9, self.center_frequencies, atol=self.fstep/2.)
            else:
                self.mask = np.ones(len(self.center_frequencies), bool)

        return

    def setup(self):

        if self.instrument=='pixie':
            self.set_frequencies()
            self.noise = self.pixie_sensitivity()
        
        elif self.instrument=='pixie2024':
            self.noise = self.pixie_sensitivity_2024()
            # print(self.center_frequencies)
            # print(self.noise)
        elif self.instrument=='firas':
            if self.low_or_high=="both":
                self.center_frequencies_low, self.noise_inv_low,self.center_frequencies_high, self.noise_inv_high =self.firas_sensitivity()
            else:
                self.center_frequencies, self.noise_inv=self.firas_sensitivity()
        else:
            sys.exit("pick 'firas' or 'pixie' as instrument")
        return

    def run_fisher_calculation(self):

        N = len(self.args)
        
        #calculate Fisher for two frequency ranges
        if self.instrument=='firas' and self.low_or_high=="both":
            
            self.center_frequencies=copy.deepcopy(self.center_frequencies_low)
            self.noise_inv=copy.deepcopy(self.noise_inv_low)
            Flow = self.calculate_fisher_matrix()

            self.center_frequencies=copy.deepcopy(self.center_frequencies_high)
            self.noise_inv=copy.deepcopy(self.noise_inv_high)
            Fhigh = self.calculate_fisher_matrix()
            
            F=Flow+Fhigh
        #pixie or just low frequencies for FIRAS
        else:
            F = self.calculate_fisher_matrix()
        
        #take into account Gaussian priors & invert Fisher
        for k in self.priors.keys():
            if k in self.args and self.priors[k] > 0:
                kindex = np.where(self.args == k)[0][0]
                F[kindex, kindex] += 1. / (self.priors[k] * self.argvals[k])**2
        normF = np.zeros([N, N], dtype=ndp)
        for k in range(N):
            normF[k, k] = 1. / F[k, k]
        self.cov = ((np.mat(normF, dtype=ndp) * np.mat(F, dtype=ndp)).I * np.mat(normF, dtype=ndp)).astype(ndp)
        self.F = F
        self.get_errors()

        return

    def get_errors(self):
        self.errors = {}
        for k, arg in enumerate(self.args):
            self.errors[arg] = np.sqrt(self.cov[k,k])
        return

    def print_errors(self, args=None):
        if not args:
            args = self.args
        for arg in args:
            #print arg, self.errors[arg], self.argvals[arg]/self.errors[arg]
            print(arg, self.argvals[arg]/self.errors[arg])

    def set_signals(self, fncs=None):
        if fncs is None:
            fncs = [sd.DeltaI_mu, sd.DeltaI_reltSZ_2param_yweight, sd.DeltaI_DeltaT,
                    fg.thermal_dust_rad, fg.cib_rad, fg.jens_freefree_rad,
                    fg.jens_synch_rad, fg.spinning_dust, fg.co_rad]
        self.signals = fncs
        
        #fiducial
        if len(self.arg_dict)==0:
            self.args, self.p0, self.argvals = self.get_function_args()
        #if parameter dictionary was specified 
        else:
            self.args, self.p0, self.argvals = self.get_function_args_custom()
        
        return

    def set_frequencies(self):
        if self.bandpass:
            self.band_frequencies, self.center_frequencies, self.binstep = self.band_averaging_frequencies()
        else:
            self.center_frequencies = np.arange(self.fmin + self.fstep/2., \
                                                self.fmax + self.fstep, self.fstep, dtype=ndp)[self.drop:]
        return

    def band_averaging_frequencies(self):
        #freqs = np.arange(self.fmin + self.bandpass_step/2., self.fmax + self.fstep, self.bandpass_step, dtype=ndp)
        freqs = np.arange(self.fmin + self.bandpass_step/2., self.fmax + self.bandpass_step/2. + self.fmin, self.bandpass_step, dtype=ndp)
        binstep = int(self.fstep / self.bandpass_step)
        freqs = freqs[self.drop * binstep : int((len(freqs) / binstep) * binstep)]
        centerfreqs = freqs.reshape((int(len(freqs) / binstep), binstep)).mean(axis=1)
        #self.windowfnc = np.sinc((np.arange(binstep)-(binstep/2-1))/float(binstep))
        return freqs, centerfreqs, binstep

    def pixie_sensitivity(self):
        sdata = np.loadtxt(this_dir+'/templates/Sensitivities.dat', dtype=ndp)
        fs = sdata[:, 0] * 1e9
        sens = sdata[:, 1]
        template = interpolate.interp1d(np.log10(fs), np.log10(sens), bounds_error=False, fill_value="extrapolate")
        skysr = 4. * np.pi * (180. / np.pi) ** 2 * self.fsky
        if self.bandpass:
            N = len(self.band_frequencies)
            noise = 10. ** template(np.log10(self.band_frequencies)) / np.sqrt(skysr) * np.sqrt(15. / self.duration) * self.mult * 1.e26
            return (noise.reshape((int( N / self.binstep), self.binstep)).mean(axis=1)).astype(ndp)
        else:
            return (10. ** template(np.log10(self.center_frequencies)) / np.sqrt(skysr) * np.sqrt(15. / self.duration) * self.mult * 1.e26).astype(ndp)

    def pixie_sensitivity_2024(self):
        
        sdata = np.loadtxt(this_dir+'/templates/pixie_mission_noise_hill.txt', dtype=ndp)    
        high_freq_cut=np.where(sdata[:, 0] * 1e9 <self.fmax)[0]
        self.center_frequencies= sdata[high_freq_cut, 0]* 1e9
        sens = sdata[high_freq_cut, 1] #jy/sr
        
        # plt.figure()
        # plt.plot(fs, sens, label='template')

        # template = interpolate.interp1d(np.log10(fs), np.log10(sens), bounds_error=False, fill_value="extrapolate")        
        # if self.bandpass:
        #     N = len(self.band_frequencies)
        #     noise = 10. ** template(np.log10(self.band_frequencies))
        #     return (noise.reshape((int( N / self.binstep), self.binstep)).mean(axis=1)).astype(ndp)
        # else:
            # plt.plot(self.center_frequencies, (10. ** template(np.log10(self.center_frequencies))).astype(ndp), label='actual noise')
            # plt.legend()
            # plt.xscale('log')
            # plt.savefig("test_interp.pdf")
            # sys.exit(0)
        return (sens).astype(ndp)

    def firas_sensitivity(self):

        #using monopole errors
        if self.file_type=='monopole':
            data=np.load(this_dir+'/data/'+self.fname, allow_pickle=True)
            if self.low_or_high=="both":

                data_dict_high=prepare_data_highf_masked_nolines(fname=self.fname,sky_frac=self.fsky,method=self.method, cutoff_freq=self.highf_thresh, ind_mask=self.highf_mask)
                data_dict_low=prepare_data_lowf_masked_nolines(fname=self.fname,sky_frac=self.fsky,method=self.method, ind_mask=self.lowf_mask)

                return data_dict_low['freqs'], data_dict_low['cov_inv'], data_dict_high['freqs'], data_dict_high['cov_inv']

            elif self.low_or_high=="lowf":

                data_dict=prepare_data_lowf_masked_nolines(fname=self.fname,sky_frac=self.fsky,method=self.method, ind_mask=self.lowf_mask)

                return data_dict['freqs'], data_dict['cov_inv']

        #using covariance file
        elif self.file_type=='noise':

            freqs, tot, C, beta, JCJ, PEP, PUP, PTP=np.loadtxt(this_dir+'/data/'+self.fname, unpack=True)

            if self.low_or_high=='lowf':
                max_freq=640
                mask_ind=np.where(freqs[self.lowf_mask[0]:self.lowf_mask[-1]]<max_freq)[0]
            else:
                mask_ind=np.where(freqs<self.highf_thresh)[0]

            err=np.zeros(len(freqs))

            if self.which_noise=='tot':
                err=np.copy(np.abs(tot))/np.sqrt(N_pixels)

            if 'C' in self.which_noise:
                #print("including C")
                skysr = 4. * np.pi * (180. / np.pi) ** 2 * self.fsky
                err+=np.abs(C)/np.sqrt(skysr)*C_extra_factor

            if 'beta' in self.which_noise:
                #print("including beta")
                err+=np.abs(beta)/np.sqrt(N_pixels)

            if 'JCJ' in self.which_noise:
                #print("including JCJ")
                err+=np.abs(JCJ)/np.sqrt(N_pixels)

            if 'PEP' in self.which_noise:
                #print("including PEP")
                err+=np.abs(PEP)/np.sqrt(N_pixels)

            if 'PUP' in self.which_noise:
                #print("including PUP")
                err+=np.abs(PUP)/np.sqrt(N_pixels)

            if 'PTP' in self.which_noise:
                #print("including PTP")
                err+=np.abs(PTP)/np.sqrt(N_pixels)

            masked_freqs=freqs[mask_ind]*1e9
            masked_err=err[mask_ind]*1.e6

            if self.remove_lines==True:

                outliers=remove_lines(masked_freqs,1.0)
                masked_freqs=np.delete(masked_freqs, outliers)
                masked_err=np.delete(masked_err, outliers)

            return masked_freqs, masked_err

    def get_function_args(self):
        targs = []
        tp0 = []
        for fnc in self.signals:
            argsp = inspect.getargspec(fnc)
            args = argsp[0][1:]
            p0 = argsp[-1]
            targs = np.concatenate([targs, args])
            tp0 = np.concatenate([tp0, p0])
        return targs, tp0, dict(zip(targs, tp0))

    #get parameters if specified
    def get_function_args_custom(self):
        targs = []
        tp0 = []
        for key,value in self.arg_dict.items():

            targs.append(key)
            tp0.append(value)

        targs=np.array(targs)
        tp0=np.array(tp0)

        return targs, tp0, dict(zip(targs, tp0))

    def calculate_fisher_matrix(self):

        if self.instrument=='firas':
            #noise is not a diagonal matrix
            N = len(self.p0)
            F = np.zeros([N, N], dtype=ndp)
            for i in range(N):
                dfdpi = self.signal_derivative(self.args[i], self.p0[i])
                first_term=np.dot(dfdpi, self.noise_inv)
                for j in range(N):
                    dfdpj = self.signal_derivative(self.args[j], self.p0[j])
                    F[i, j] = np.dot(first_term, dfdpj)

        elif self.instrument=='pixie' or self.instrument=='pixie2024':
            N = len(self.p0)
            F = np.zeros([N, N], dtype=ndp)
            for i in range(N):
                dfdpi = self.signal_derivative(self.args[i], self.p0[i])
                dfdpi /= self.noise
                for j in range(N):
                    dfdpj = self.signal_derivative(self.args[j], self.p0[j])
                    dfdpj /= self.noise
                    #F[i, j] = np.dot(dfdpi, dfdpj)
                    F[i, j] = np.dot(dfdpi[self.mask], dfdpj[self.mask])

        return F

    def signal_derivative(self, x, x0):
        h = 1.e-4
        zp = 1. + h
        deriv = (self.measure_signal(**{x: x0 * zp}) - self.measure_signal(**{x: x0})) / (h * x0)
        return deriv

    def measure_signal(self, **kwarg):

        if self.bandpass:
            frequencies = self.band_frequencies
        else:
            frequencies = self.center_frequencies

        N = len(frequencies)
        model = np.zeros(N, dtype=ndp)
        for fnc in self.signals:
            argsp = inspect.getargspec(fnc)
            args = argsp[0][1:]
            if len(kwarg) and list(kwarg.keys())[0] in args:
                model += fnc(frequencies, **kwarg)
        if self.bandpass:
            #rmodel = model.reshape((N / self.binstep, self.binstep))
            #total = rmodel * self.windowfnc
            return model.reshape((int(N / self.binstep), self.binstep)).mean(axis=1)
            #return total.mean(axis=1)
        else:
            return model
