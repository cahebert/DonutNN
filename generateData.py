import numpy as np
import galsim
import random as rnd
import pandas as pd

d_lsst = 8.4  #diameter of aperture (LSST)
ref_lam = 700  #reference wavelength
Z_N_max = 15  #highest Zernike order to generate (naming starts at 2)
N = 1000  #number of donut-wavefront pairs to generate

#df = pd.DataFrame(columns=['psf','wavefront','aberrations'])
dfList = []

for i in range(N):
    df = pd.DataFrame(columns=['psf','wavefront','aberrations'])
    ## generate random aberrations, symmetric around zero, max amplitude .2
    amplitudes = [(rnd.random()-0.5)*.4 for i in range(2,Z_N_max)]
    amplitudes.insert(0,0), amplitudes.insert(0,0)   ## insert 0 for first entry (unused) and second (piston)
    
    ## make optical wavefront (don't incude Z4)
    z_poly = galsim.OpticalScreen(diam=d_lsst, aberrations=amplitudes,lam_0=ref_lam)
    aperture = galsim.Aperture(diam=d_lsst, lam=ref_lam, obscuration=.6, screen_list=z_poly,
                           oversampling=.5,pad_factor=.25)
    ## make wavefront, only keep what comes through the telescope aperture 
    wf = z_poly.wavefront(aperture.u,aperture.v)*aperture.illuminated 
    wavefront = wf/ref_lam ## now in waves!

    ## set defocus for to make the donut 
    amplitudes[4] = 32   
    
    ## make optical psf
    optics_psf = galsim.OpticalPSF(lam=ref_lam, diam=d_lsst, obscuration=0.61,
                               oversampling=.5, pad_factor=4.0,
                               aberrations=amplitudes)
    kolmogorov = galsim.Kolmogorov(lam=ref_lam, r0 = .2)
    psf = galsim.Convolve(optics_psf,kolmogorov).drawImage(nx=100,ny=100, scale=0.2).array
    optics_psf = psf/psf.max() #normalize pixels


    dfList.append(df.append({'psf': optics_psf,'wavefront': wavefront, 'aberrations':amplitudes},ignore_index=True))
    
data = pd.concat(dfList,ignore_index=True)

data.to_pickle('/Users/clairealice/Documents/Research/Burchat/DonutNN/simulatedData.p')
