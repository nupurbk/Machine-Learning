# -*- coding: utf-8 -*-
"""
Name :Nupur Kulkarni
Machine learning Assignment 1, part 3
Data Set : heart.data


Useful Features :
Thal(13)
Thalach(8)-maximum heart rate

Limited Use features :
Fasting blood sugur(FBS)(6)
RestEcg(7)


"""

import numpy as np

import pylab as pl


header_row = ['label','Max Heart rate','Thal','Cholestoral','FBS']
data = np.genfromtxt('heart.data', delimiter=',',usecols=(1,9,14,7,8),names=header_row,comments='#',dtype=[float for n in range(5)])
 

from matplotlib import pyplot as plt
 
 
#dictionary for positive data
dp=dict()
#dictionary for negative data
dn=dict()
for m in range(1,5):
    positive =[]
    negative=[]
    for i in range(len(data)):
        if data[i][0]==1.0:
            positive.append(data[i][m])
        else:
            negative.append(data[i][m])
     
    dp[m]=positive
    dn[m]=negative

    
plt.figure(figsize=(100,60))
f, axarr = plt.subplots(2,2)
plt.subplots_adjust(hspace=0.75)
axarr[0, 0].hist(dp[1],5, normed=True, facecolor='b', ls='dashed',alpha=0.75,label='Positive', cumulative=0)
axarr[0, 0].hist(dn[1],5, normed=True, facecolor='r',alpha=0.5,label='Negative', cumulative=0)
axarr[0, 0].set_title('Plot 1 :Max Heart Rate')

axarr[0, 1].hist(dp[2],5, normed=True, facecolor='b', ls='dashed',alpha=0.75,label='Positive', cumulative=0)
axarr[0, 1].hist(dn[2],5, normed=True, facecolor='r', ls='dashed',alpha=0.5,label='Negative', cumulative=0)
axarr[0, 1].set_title('Plot 2:Thal')

axarr[1, 0].hist(dp[3],5, normed=True, facecolor='b', ls='dashed',alpha=0.75,label='Positive', cumulative=0)
axarr[1, 0].hist(dn[3],5, normed=True, facecolor='r', ls='dashed',alpha=0.5,label='Negative', cumulative=0)
axarr[1, 0].set_title('Plot 3:FBS')

axarr[1, 1].hist(dp[4],5, normed=True, facecolor='b', ls='dashed',alpha=0.75,label='Positive', cumulative=0)
axarr[1, 1].hist(dn[4],5, normed=True, facecolor='r', ls='dashed',alpha=0.5,label='Negative', cumulative=0)
axarr[1, 1].set_title('Plot 4:Resting ECG')

#plt.figlegend(,("Positive","Negative"), loc = 'upper right', ncol=5, labelspacing=0. )
plt.tight_layout()
plt.savefig('out.pdf',dpi=200)
plt.show()



