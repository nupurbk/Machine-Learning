
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 13:11:15 2016

@author: Nupur

Perceptron with bias on Heart data
"""
import time
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cross_validation import train_test_split

#heartdata



class Perceptron :
    """
    An implementation of the perceptron algorithm.
    Note that this implementation does not include a bias term"""
 
    def __init__(self, max_iterations=100, learning_rate=0.2) :
 
        self.max_iterations = max_iterations
        self.learning_rate = learning_rate
 
    def fit(self, X, y) :
        
        self.w = np.zeros(len(X[0]))
        converged = False
        iterations = 0
        self.b=0
        count=0
        while (not converged and iterations < self.max_iterations) :
            converged = True
            for i in range(len(X)) :
                if y[i] *(self.discriminant(X[i])) <= 0 :
                    self.w = self.w + y[i] * self.learning_rate * X[i]+ self.b
                    self.b = self.b + y[i] * self.learning_rate
                    converged = False
            
            #plot_data(X, y, self.w)
            count+=1
            iterations += 1
        self.converged = converged
       
        if converged :
            print ('converged in %d iterations ' % iterations)
        else:
            print("Data set is linearly inseperable")
            
        print("In sample Error for Pocket (Ein) :", float(count)/len(X))
 
    def discriminant(self, x) :
        return np.inner(self.w, x)
 
    def predict(self, X) :
        scores = np.inner(self.w, X)
        return np.sign(scores+self.b)
        
        
    def test(self,X,y):
        count=0.0
        eOut=0.0
        for j in range(len(X)):
            if self.predict(X[j])!= y[j]:
                count=count+1
        eOut=count/len(X)
        print("Misclassified sample :",count)
        print("length of sample :",len(X))
        print("Eout",eOut)
        return eOut
            
def generate_separable_data() :
    X= np.genfromtxt("heart.data", delimiter=",",usecols=range(2,15),comments="#")
    y= np.genfromtxt("heart.data", delimiter=",",usecols=(1),comments="#")
    X=np.array(X)
    print (X,X.shape)
    #y= np.genfromtxt("gisette_train.label",delimiter=" ", comments="#")
    y=np.array(y)
    rows=len(X)
    s=int(0.60 *rows)+1
    
    #Se g r e g a t i on to t r a i n i n g and t e s t i n g data
    X_train=X[0:s]
    X_test=X[s:rows]
    y_train=y[0:s]
    y_test=y[s:rows]
    return X_train,X_test,y_train,y_test
 
def plot_data(X, y, w) :
    fig = plt.figure(figsize=(5,5))
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    a = -w[0]/w[1]
    pts = np.linspace(-1,1)
    plt.plot(pts, a*pts, 'k-')
    cols = {1: 'r', -1: 'b'}
    for i in range(len(X)): 
        plt.plot(X[i][0], X[i][1], cols[y[i]]+'o')
    plt.show()
    
    

 
if __name__=='__main__' :
    
    
    start_time = time.time()
   # X,y,w = generate_separable_data(40)
    X= np.genfromtxt("heart.data", delimiter=",",usecols=range(2,15),comments="#")
    y= np.genfromtxt("heart.data", delimiter=",",usecols=(1),comments="#")
#qsar data

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,random_state=0)
    #X_train, X_test, y_train, y_test=generate_separable_data()

   
    p = Perceptron()
    p.fit(X_train,y_train)
    p.test(X_test,y_test)
    
   
    print("--- %s seconds ---" % (time.time() - start_time))
