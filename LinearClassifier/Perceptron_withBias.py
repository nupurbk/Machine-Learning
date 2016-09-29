
"""
Created on Tue Sep 13 13:11:15 2016

@author: Nupur
Perceptron with bias for QSAR and Gisettte with 10 random splits and average error
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
        """
        Train a classifier using the perceptron training algorithm.
        After training the attribute 'w' will contain the perceptron weight vector.
 
        Parameters
        ----------
 
        X : ndarray, shape (num_examples, n_features)
        Training data.
 
        y : ndarray, shape (n_examples,)
        Array of labels.
 
        """
        self.w = np.zeros(len(X[0]))
        converged = False
        iterations = 0
        self.b=0
        count=0
        while (not converged and iterations < self.max_iterations) :
            converged = True
            for i in range(len(X)) :
                if y[i] * self.discriminant(X[i]) <= 0 :
                    self.w = self.w + y[i] * self.learning_rate * X[i]+ self.b
                    self.b = self.b + y[i] * self.learning_rate
                    converged = False
                    #plot_data(X, y, self.w)
            iterations += 1
            count+=1
        self.converged = converged
       
        if converged:
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
        print("Eout :",eOut)
        return eOut
            
def generate_separable_data() :
    X= np.genfromtxt("biodeg.data", delimiter=";",usecols=np.arange(0,41))
    X=np.array(X)
   
    y1= np.genfromtxt("biodeg.data", delimiter=";",usecols=41,dtype=str)
    y1=np.array(y1)
    
   
    
    y=[]
    for i in range(0,len(y1)):
        l=assignlabels(y1[i])
        y.append(l)
    
    
    
    X1= np.genfromtxt("gisette_train.data",delimiter=" ", comments="#")
    X1=np.array(X1)
   
    y2= np.genfromtxt("gisette_train.label",delimiter=" ", comments="#")
    y2=np.array(y2)
    
    """
    rows=len(X)
    s=int(0.60 *rows)+1
    
    #Se g r e g a t i on to t r a i n i n g and t e s t i n g data
    X_train=X[0:s]
    X_test=X[s:rows]
    y_train=y[0:s]
    y_test=y[s:rows]
    
    print("Train data",X_train.shape)
    #print("label train data",y_train.shape)
    print("Test data",X_test.shape)
    #print("label test data",y_test.shape)
    
    return X_train,X_test,y_train,y_test
    """
    return X1,y2,X,y
    
def assignlabels(l):
    if l=='RB':
        return 1
    else:
        return -1
 
 
if __name__=='__main__' :
    
    
    start_time = time.time()
    Xg,yg,Xq,yq= generate_separable_data()
    print("Perceptron for Gisette data : ")
    avgerror=[]
    for i in range (0,10):
        X_train, X_test, y_train, y_test = train_test_split(Xg, yg, test_size=0.4, random_state=i)
        #X_train, X_test, y_train, y_test =generate_separable_data()
        p = Perceptron()
        p.fit(X_train,y_train)
        Eout=p.test(X_test,y_test)
        avgerror.append(Eout)
        print("Eout for",i," iteration :",Eout)
    averageEout=sum(avgerror)/10
    print("Average  Eout error for perceptron with bias for GIsette :",averageEout)
    #standard deviation 
    stddev=0
    stddev=np.std(avgerror)
    print("Standard Deviation for perceptron with bias gisette :",stddev)
    
    
     
    #for QSAR data
    print("Perceptron for QSAR data : ")
    
    avgerror=[]
    for i in range (0,10):
        X1_train, X1_test, y1_train, y1_test = train_test_split(Xq, yq, test_size=0.4, random_state=i)
        #X_train, X_test, y_train, y_test =generate_separable_data()
        p = Perceptron()
        p.fit(X1_train,y1_train)
        Eout=p.test(X1_test,y1_test)
        avgerror.append(Eout)
        print("Eout for",i," iteration :",Eout)
    averageEout=sum(avgerror)/10
    print("Average  Eout error for perceptron with bias QSAR:",averageEout)
    #standard deviation 
    stddev=0
    stddev=np.std(avgerror)
    print("Standard Deviation for perceptron with bias QSAR:",stddev)
    
    
   
    
    print("--- %s seconds ---" % (time.time() - start_time))
