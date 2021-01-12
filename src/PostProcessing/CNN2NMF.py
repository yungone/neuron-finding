import thunder as td
from extraction import NMF
from registration import CrossCorr
import numpy as np
import os
import json
from PIL import Image


class Parameters:
    def __init__(self):
        self.baseDirectory = '/csphome/jiahaoxu/Jerry/CS/8360/outputs/filtered_outputs/best_outputs/'
        self.prefix = 'neurofinder.'

        self.datasets = [ 
                        '00.00',
                        '00.01',  
                        '01.00',  
                        '01.01',  
                        '02.00',  
                        '02.01',  
                        '03.00',  
                        '04.00',  
                        '04.01'
                        ]

        self.medianFilterSize = 2
        self.gaussianFilterSigma = 1

        self.nmfNumComp = 5
        self.nmfMaxIter = 50
        self.nmfMaxSize = 'full'
        self.nmfMinSize = 10
        self.nmfPercent = 99
        self.nmfOverlap = 0.1

        self.nmfFitChunkSize = 50
        self.nmfFitPadding = 15

        self.nmfMergeOverlap = 0.1
        self.nmfMergeMaxIter = 10
        self.nmfMergeKNN = 20


def main():
    para = Parameters()
    base = para.baseDirectory + '/' + para.prefix

    submission = []

    for dataset in para.datasets:
        dataset = dataset + '.test'
        path = base + dataset
        print(path)
        
        b = Image.open(path + '.tiff')
        a = np.zeros((100, 512, 512))
        for i in range(100):
            a[i] = np.array(b)
        
        print(a.shape, np.count_nonzero(a), np.min(a), np.max(a))
        
        algorithm = NMF(k=para.nmfNumComp, 
                        max_iter=para.nmfMaxIter, 
                        max_size=para.nmfMaxSize,
                        min_size=para.nmfMinSize,
                        percentile=para.nmfPercent, 
                        overlap=para.nmfOverlap)

        model = algorithm.fit(a, 
                              chunk_size=(para.nmfFitChunkSize, para.nmfFitChunkSize), 
                              padding=(para.nmfFitPadding, para.nmfFitPadding))

        merged = model.merge(overlap=para.nmfMergeOverlap,
                             max_iter=para.nmfMergeMaxIter,
                             k_nearest=para.nmfMergeKNN)

        regions = [{'coordinates': region.coordinates.tolist()} for region in merged.regions]
        result = {'dataset': dataset, 'regions': regions}
        submission.append(result)

    with open('submission.json', 'w') as f:
        f.write(json.dumps(submission))


if __name__ == '__main__':
    main()
