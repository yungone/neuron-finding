import thunder as td
from extraction import NMF
from registration import CrossCorr
import numpy as np
import os
import json
import preprocess


def preprocess(input, para):
    """
    @param: input: type, array like, input array to be filtered
    @param: para: type, object, an object that contains parsed arguments
            medianFilterSize = 2
            gaussianFilterSigma = 1
    return: type, same shape as input, medianFilter
    """  
    # median filter
    dataMF = input.median_filter(size=para.medianFilterSize)

    # gaussian filter
    dataGF = dataMF.gaussian_filter(sigma=para.gaussianFilterSigma)

    # We can assume the neurons are motionless
    # registration
    # algorithmReg = CrossCorr()
    # modelReg = algorithmReg.fit(dataGF, reference=dataGF.mean().toarray())
    # registered = modelReg.transform(dataGF)
    
    return dataGF