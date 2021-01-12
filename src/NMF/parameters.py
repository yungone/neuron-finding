class Parameters:
    def __init__(self):
        self.baseDirectory = '/Users/jerryhui/Downloads/project3_neurofinder.all.test'
        self.prefix = 'neurofinder.'
        self.datasets = [   
                        '00.00'
                        ]

        # self.datasets = [
        #                 '00.00',  
        #                 '01.00',  
        #                 '02.00',  
        #                 '03.00',  
        #                 '04.01',
        #                 '00.01',  
        #                 '01.01',  
        #                 '02.01',  
        #                 '04.00'
        #                 ]

        self.medianFilterSize = 2
        self.gaussianFilterSigma = 1

        self.nmfNumComp = 5
        self.nmfMaxIter = 10
        self.nmfMaxSize = 'full'
        self.nmfMinSize = 10
        self.nmfPercent = 99
        self.nmfOverlap = 0.1

        self.nmfFitChunkSize = 30
        self.nmfFitPadding = 25

        self.nmfMergeOverlap = 0.1
        self.nmfMergeMaxIter = 10
        self.nmfMergeKNN = 20