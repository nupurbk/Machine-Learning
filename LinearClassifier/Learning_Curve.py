# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 16:12:51 2016

@author: Nupur
"""

import numpy as np
from matplotlib import pyplot as plt


trainingsamples=[4,16,32,64,256,1024,2042,3601]
misclassified_Samples=[1119,526,436,428,222,116,99,32]



outof_sampleerror=np.zeros(len(misclassified_Samples))
for i in range(0,len(trainingsamples)):
        outof_sampleerror[i]=float(misclassified_Samples[i])/2399

plt.plot(trainingsamples,outof_sampleerror,linestyle="-", color="b", label="E_out",marker="o")
plt.grid(True)

plt.xscale("log",basex=4)

plt.xlabel("Number of Training Examples [log]", fontsize = 14)

plt.ylabel("Mislasified Examples [Eout]", fontsize = 14)

plt.title("Learning Curve for Gisette Testing Dataset")

plt.legend()



plt.savefig('out.pdf',dpi=200)

plt.show()
