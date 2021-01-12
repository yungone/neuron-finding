"""
This code is highly inspired by 
https://gist.github.com/freeman-lab/330183fdb0ea7f4103deddc9fae18113
https://gist.github.com/freeman-lab/3e98ea4ccb96653c8fb085090079ce21

pip install thunder-python
pip install thunder-extraction
pip install thunder-registration
"""

import thunder as td
from extraction import NMF
from registration import CrossCorr
import numpy as np
import os
import json
import parameters 
from preprocess import preprocess

DEBUG = True    

def getTestResults(para):
    """
    gerenerate the testing results
    
    @param: para: type, object, an object that contains parsed arguments
            baseDirectory = "/Users/jerryhui/Downloads/project3_neurofinder.all.test"
            prefix = "neurofinder."
            datasets = ["00.00"]

            medianFilterSize = 2
            gaussianFilterSigma = 1

            nmfNumComp = 5
            nmfMaxIter = 20
            nmfMaxSize = "full"
            nmfMinSize = 20
            nmfPercent = 95
            nmfOverlap = 0.1

            nmfFitChunkSize = 50
            nmfFitPadding = 25

            nmfMergeOverlap = 0.5
            nmfMergeMaxIter = 2
            nmfMergeKNN = 1
    return: void
    """
    base = para.baseDirectory + '/' + para.prefix
    # submission = []

    for dataset in para.datasets:
        dataset = dataset + '.test'
        path = os.path.join(base + dataset, 'images')
        
        data = td.images.fromtif(path, stop=None, ext='tiff')
        if DEBUG:
            data = data[::10,:,:]
        print(data.shape)

        # data preprocessing
        data2 = preprocess(data, para)

        algorithm = NMF(k=para.nmfNumComp, 
                        max_iter=para.nmfMaxIter, 
                        max_size=para.nmfMaxSize,
                        min_size=para.nmfMinSize,
                        percentile=para.nmfPercent, 
                        overlap=para.nmfOverlap)

        model = algorithm.fit(data2, 
                              chunk_size=(para.nmfFitChunkSize, para.nmfFitChunkSize), 
                              padding=(para.nmfFitPadding, para.nmfFitPadding))

        merged = model.merge(overlap=para.nmfMergeOverlap,
                             max_iter=para.nmfMergeMaxIter,
                             k_nearest=para.nmfMergeKNN)
        
        # generate resultant regions
        regions = [{'coordinates': region.coordinates.tolist()} for region in merged.regions]
        result = {'dataset': dataset, 'regions': regions}
        submission = [result]
        print('%s, %s chunk size, has been finished' % (dataset, para.nmfFitChunkSize))

        with open('submission_' + dataset + '_chunk_' + str(para.nmfFitChunkSize) + '.json', 'w') as f:
            f.write(json.dumps(submission))


if __name__ == '__main__':
    para = parameters.Parameters()
    getTestResults(para)
