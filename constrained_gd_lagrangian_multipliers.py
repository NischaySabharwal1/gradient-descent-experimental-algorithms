import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn import datasets  # for dummy data
from dataclasses import dataclass

x, y = datasets.make_blobs(n_samples=10000, centers=2, n_features=2, center_box=(-10, 10))
x = np.hstack((x, np.ones((len(x),1))))
y[y==0]=-1

@dataclass
class datapoints:
    x: np.array
    y: np.array

data = datapoints(x,y)

def plotting(w):
    w1,w2,w0 = w
    h_slope = -w1/w2
    h_intercept = -w0/w2
    x_ = np.arange(-10,10)
    plt.plot(data.x[:, 0][data.y == -1], data.x[:, 1][data.y == -1], 'g^')
    plt.plot(data.x[:, 0][data.y == 1], data.x[:, 1][data.y == 1], 'bs')
    y_ = h_slope*x_+h_intercept
    plt.plot(x_, y_)
    plt.show()


def g(w):
    return np.sqrt(np.sum(w**2))

def f(x,y,w):
    w = w.reshape(-1,1)
    return np.sum(y*np.dot(x,w))
    

def L(x,y,w,lda):
    return f(x,y,w) + lda*(g(w)-1)

def derivative(x,y,w, lda):
    xy = np.sum(np.array([x[i]*y[i] for i in range(len(x))]), axis = 0)
    l = (w*lda)/g(w)
    derw = xy + l
    derlda = g(w)-1
    return derw, derlda

def step(x,y,w, lda, L, lr):
    gradw, gradlda = derivative(x,y,w,lda)
    new_w = w - lr*gradw
    new_lda = lda - lr*gradlda
    new_lxz = L(x,y,new_w, lda)
    return new_w, new_lda, new_lxz



def gradient_descent(x,y,w, lda):
    
    new_lda = lda
    new_w = w
    new_lxz = L(x,y,new_w, new_lda)
    w_s, lda_s, lxz = [], [], []
    n_iter = 0

    while True:
    
        new_w, new_lda, new_lxz = step(x,y,new_w, new_lda, L, 0.1)
        w_s.append(new_w)
        lda_s.append(new_lda)
        lxz.append(new_lxz)
        n_iter += 1
    
        if len(lxz)>2 and abs(lxz[-1]-lxz[-2]) < 1e-5:
            break
        
        if n_iter > 350:
            print('Hard Stop: 350 iterations reached')
            break
        
        
        if n_iter%50==0:
            print(f'Last function value: {new_lxz}')
            print(f'Last w1 value: {new_w[0]}')
            print(f'Last w2 value: {new_w[1]}')
            print(f'Last w0 value: {new_w[-1]}')
            print(f'Last Lambda value: {new_lda}')
            print(f'Number of iterations: {n_iter}')
            #plotting(new_w)
        
    w_s = np.array(w_s)    
    print(f"Took {n_iter} iterations")
    print(f"Final loss: {new_lxz}")
    print(f"Optimum value of parameter w1: {w_s[-1,0]}")  
    print(f"Optimum value of parameter w2: {w_s[-1,1]}") 
    print(f"Optimum value of parameter w0: {w_s[-1,-1]}")

    return w_s[-1], new_lxz, lxz

#for i in range(10):
w = np.random.randn(3)
l = np.random.randn()
'''    indices = np.random.randint(0, 200, 30)
    x_ = data.x[indices]
    y_ = data.y[indices]
    w1,w2,w0 =  w[0], w[1], w[2]
    if i == 0:
        loss = L(x_, y_, w, l)
        print('Initial Loss:', loss)
        plotting(w)
        min = loss
        min_w = w
        w, loss,lxz = gradient_descent(x_, y_, min_w, l)
    else: 
        if loss<min:
            min = loss
            min_w = w'''
plotting(w)
indices = np.random.randint(0, 200, 100)
w_final, loss, lxz = gradient_descent(data.x[indices], data.y[indices], w, l)
plotting(w_final)
plt.plot(np.arange(len(lxz)), lxz)
plt.show()


