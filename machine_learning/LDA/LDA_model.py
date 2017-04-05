# _*_ coding: utf-8 _*_

"""

LDA_model created by xiangkun on 2017/3/28

"""

import numpy as np


class LDA:

    def __init__(self, k, v, m, alpha, beta):
        # set doc-topic, topic-word dirichlet prior parameter
        self.alpha = alpha
        self.beta = beta

        # set words, topic, document num
        self.K = k
        self.V = v
        self.M = m

        # set initial doc and topic distribution
        self.Doc_dist = np.random.rand(self.M, self.V)
        self.Topic_dist = np.random.rand(self.V, self.K)

    def load_docs(self):
        pass

