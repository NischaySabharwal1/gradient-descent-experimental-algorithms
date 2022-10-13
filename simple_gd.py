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

def plotting(x,y,w):
    w1,w2,w0 = w
    h_slope = -w1/w2
    h_intercept = -w0/w2
    x_ = np.arange(-10,10)
    plt.plot(x[:, 0][y == -1], x[:, 1][y == -1], 'g^')
    plt.plot(x[:, 0][y == 1], x[:, 1][y == 1], 'bs')
    y_ = h_slope*x_+h_intercept
    plt.plot(x_, y_)
    plt.show()


def g(w):
    return np.sqrt(np.sum(w**2))

def f(x,y,w):
    w = w.reshape(-1,1)
    return np.sum(y*np.dot(x,w))/g(w)

def derivative(x,y,w):
    xyw = np.sum(np.array([x[i]*y[i] for i in range(len(x))])*g(w), axis = 0)
    t2 = w*f(x,y,w)
    derw = ((xyw-t2)/g(w)**2)
    return derw 

def step(x,y,w, f, lr):
    gradw = derivative(x,y,w)
    new_w = w - lr*gradw
    new_lxz = f(x,y,new_w)
    return new_w, new_lxz



def gradient_descent(x,y,w):
    
    new_w = w
    new_lxz = f(x,y,new_w)
    w_s, lxz = [], []
    n_iter = 0

    while True:
    
        new_w, new_lxz = step(x,y,new_w, f, 0.1)
        w_s.append(new_w)
        lxz.append(new_lxz)
        n_iter += 1
    
        if len(lxz)>2 and abs(lxz[-1]-lxz[-2]) < 1e-5:
            break
        
        if n_iter > 100:
            print(f'Hard Stop: {100} iterations reached')
            break
        
        
        if n_iter%200==0:
            print(f'Last function value: {new_lxz}')
            print(f'Last w1 value: {new_w[0]}')
            print(f'Last w2 value: {new_w[1]}')
            print(f'Last w0 value: {new_w[-1]}')
            print(f'Number of iterations: {n_iter}')
            plotting(new_w)
        
    w_s = np.array(w_s)    
    print(f"Took {n_iter} iterations")
    print(f"Final loss: {new_lxz}")
    print(f"Optimum value of parameter w1: {w_s[-1,0]}")  
    print(f"Optimum value of parameter w2: {w_s[-1,1]}") 
    print(f"Optimum value of parameter w0: {w_s[-1,-1]}")

    return w_s[-1], new_lxz, lxz

w = np.random.randn(3)

print('Value of coefficients:', w)
plotting(w)
indices = np.random.randint(0, 200, 100)
w_final, loss, lxz = gradient_descent(data.x[indices], data.y[indices], w)
plotting(w_final)
plt.plot(np.arange(len(lxz)), lxz)
plt.show()


