![ml_5 8](https://github.com/gummadileepkumar/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118707761/49303939-55d1-4976-9f4b-1f9668c62af9)# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

  1. Import the packages required.
  2. Read the dataset.
  3. Define X and Y array.
  4. Define a function for costFunction,cost and gradient.
  5. Define a function to plot the decision boundary and predict the Regression value.

## Program:
Program to implement the the Logistic Regression Using Gradient Descent.
### Developed by: Gumma Dileep Kumar
### RegisterNumber:  212222240032
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

data=np.loadtxt("ex2data1.txt",delimiter=',')
X=data[:,[0,1]]
y=data[:,2]
print("Array of X") 
X[:5]
print("Array of y") 
y[:5]
plt.figure()
plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend()
print("Exam 1- score Graph")
plt.show()
def sigmoid(z):
    return 1/(1+np.exp(-z))
plt.plot()
X_plot=np.linspace(-10,10,100)
plt.plot(X_plot,sigmoid(X_plot))
print("Sigmoid function graph")
plt.show()
def costFunction (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
    grad=np.dot(X.T,h-y)/X.shape[0]
    return J,grad
X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
J,grad=costFunction(theta,X_train,y)
print("X_train_grad value")
print(J)
print(grad)
X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([-24,0.2,0.2])
J,grad=costFunction(theta,X_train,y)
print("Y_train_grad value")
print(J)
print(grad)
def cost (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
    return J

def gradient (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    grad=np.dot(X.T,h-y)/X.shape[0]
    return grad 
   
X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
res=optimize.minimize(fun=cost,x0=theta,args=(X_train,y),method='Newton-CG',jac=gradient)
print(" Print res.x")
print(res.fun)
print(res.x)   
def plotDecisionBoundary(theta,X,y):
    x_min,x_max=X[:,0].min()-1,X[:,0].max()+1
    y_min,y_max=X[:,1].min()-1,X[:,1].max()+1
    xx,yy=np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1))
    X_plot=np.c_[xx.ravel(),yy.ravel()]
    X_plot=np.hstack((np.ones((X_plot.shape[0],1)),X_plot))
    y_plot=np.dot(X_plot,theta).reshape(xx.shape)
    plt.figure()
    plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
    plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
    plt.contour(xx,yy,y_plot,levels=[0])
    plt.xlabel("Exam 1 score")
    plt.ylabel("Exam 2 score")
    plt.legend()
    plt.show()  
print("Decision boundary - graph for exam score")
plotDecisionBoundary(res.x,X,y)
prob=sigmoid(np.dot(np.array([1, 45, 85]),res.x))
print("Proability value ")
print(prob)
def predict(theta,X):
    X_train =np.hstack((np.ones((X.shape[0],1)),X))
    prob=sigmoid(np.dot(X_train,theta))
    return (prob>=0.5).astype(int)
print("Prediction value of mean")
np.mean(predict(res.x,X)==y)

```

## Output:
### Array of X:

![ml_5 1](https://github.com/gummadileepkumar/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118707761/13055b72-9d8c-404b-b88e-a163458f9e4a)

### Array of Y:
![ml_5 2](https://github.com/gummadileepkumar/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118707761/84ea5d29-9db8-4543-8ca2-ab611fbdf51d)


### Score Graph:
![ml_5 3](https://github.com/gummadileepkumar/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118707761/c48485fe-8d74-4234-93d9-489709a493b7)


### Sigmoid Function Graph:
![ml_5 4](https://github.com/gummadileepkumar/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118707761/7bdc1d15-60a1-436e-ad8a-41fede916d95)


### X_train_grad Value:
![ml_5 5](https://github.com/gummadileepkumar/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118707761/2d876491-844a-4bf6-af54-37006ebaf45b)

### Y_train_grad Value:
![ml_5 6](https://github.com/gummadileepkumar/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118707761/2da95d30-a01e-4503-950c-c1b2c392e904)


### Print res_X:
![ml_5 7](https://github.com/gummadileepkumar/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118707761/f1072a53-df3b-40b3-938d-8ed42e337921)



### Decision boundary:
![ml_5 8](https://github.com/gummadileepkumar/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118707761/41b59584-0621-4b3b-908f-dee77657daed)


### Probability Value:
![ml_5 9](https://github.com/gummadileepkumar/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118707761/e2e2af67-9d55-476c-b615-746c82151bcc)


### Prediction Value of Mean:
![ml_5 10](https://github.com/gummadileepkumar/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118707761/f907c7e2-6796-484a-8106-a173b7d7ac46)


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.
