Problem 1 (Practice of scalar‐based backpropagation).
 You are required to calculate the gradients of  f(x,w) = 1/(2 + (torch.sin(x1*w1)**2) + torch.cos(x2*w2))
(a) Use computational graph for calculation 
(b) Based on(a), write a program to implement the computational graph 
and verify your answer in (a). 



Problem 2 (Practice of vector‐based backpropagation). 
You are required to calculate the gradients of f(X,W) = ||sigma(X,W)||^2 with respect to xi and Wi,j. 
Here ‖∙‖ଶ is the calculation of L2 loss, W is 3‐by‐3 matrix and x is 3‐by‐1 
vector, and 𝜎ሺ∙ሻ is sigmoid function that performs element‐wise sigmoid 
operation.  
(a) Use computational graph for calculation 
(b) Based on(a), write a program to implement the computational graph 
and verify your answer in (a). You can use vectorized approach to 
simply your codes. 

