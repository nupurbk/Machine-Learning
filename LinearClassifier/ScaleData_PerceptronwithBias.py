
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 01:39:50 2016

@author: Nupur

Perceptron with bias on scaled Heart data
"""
import time
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cross_validation import train_test_split


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
             
            count+=1#plot_data(X, y, self.w)
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
        return np.sign(scores)
        
     
   
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
    
 
    X_norm=norm(X)
        #y_norm=norm(y)
    print(X_norm)
    rows=len(X_norm)
    s=int(0.60 *rows)+1
    
    #Se g r e g a t i on to t r a i n i n g and t e s t i n g data
    X_train=X_norm[0:s]
    X_test=X_norm[s:rows]
    y_train=y[0:s]
    y_test=y[s:rows]
    return X_train,X_test,y_train,y_test
 

def norm(X):
    X=np.array(X)
    norm_array=np.zeros(X.shape)
    
    a=-1
    b=1

    for i in range(0,len(X[0])):
        #find minimum and maximum of each feature
        for j in range(len(X)):
            minimum=np.amin(X[i])
            maximum=np.amax(X[i])
           
        for j in range(len(X)):
            norm_array[j][i]=(((b-a)*(X[j][i]-minimum))/(maximum-minimum))+a           
        
    min2=np.amin(norm_array)
    max2=np.amax(norm_array)

    print("Scaled heart data :",norm_array)
    print("min 2 :",min2)
    print("max2 :",max2)    
    return norm_array
    
    

 
if __name__=='__main__' :
        
        #Heart Data
        X= np.genfromtxt("heart.data", delimiter=",",usecols=range(2,15),comments="#")
        y= np.genfromtxt("heart.data", delimiter=",",usecols=(1),comments="#")
         
        X=np.array(X)
        
        #Normalization        
        X=norm(X)
        
              
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,random_state=0)
        

       
        
        #X_train, X_test, y_train, y_test=generate_separable_data()
        p = Perceptron()
       
       
        p.fit(X_train,y_train)
       
        p.test(X_test,y_test)
      
   
        #print("--- %s seconds ---" % (time.time() - start_time))
