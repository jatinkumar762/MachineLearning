# ES654-2020 Assignment 3

*Jatin Kumar* - *19210045*

------

> Write the answers for the subjective questions here

Assume N  training examples and D features

#Nomral Equation
O(D*D*N)  to multiply XT by X
O(D*N) to multiply XT by Y
O(D*D*D) to compute (XTX)−1(XTY)
assume that N>D 
Therefore the total time complexity is  O(D*D*N)

#Gradient descent
O(N*D) for computing X*Theta
O(N) for computing X*Theta - Y
O(N*D) for computing alpha*XT
O(N*D) for computing alpha*XT(X*Theta - Y)
O(D) for computing theta = theta - alpha*XT(X*Theta - Y)
Overall time complexity is O(N*D*t)


Using gradient: 0.04459100000000005
Using Normal Equation 0.15008529999999976


