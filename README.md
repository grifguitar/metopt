# metopt

![image.png](https://github.com/grifguitar/metopt/blob/main/img/newplot.png?raw=true)
![image.png](https://github.com/grifguitar/metopt/blob/main/img/newplot1.png?raw=true)
![image.png](https://github.com/grifguitar/metopt/blob/main/img/newplot2.png?raw=true)
![image.png](https://github.com/grifguitar/metopt/blob/main/img/newplot3.png?raw=true)
![image.png](https://github.com/grifguitar/metopt/blob/main/img/newplot4.png?raw=true)
![image.png](https://github.com/grifguitar/metopt/blob/main/img/newplot5.png?raw=true)
![image.png](https://github.com/grifguitar/metopt/blob/main/img/newplot6.png?raw=true)
![image.png](https://github.com/grifguitar/metopt/blob/main/img/newplot7.png?raw=true)
![image.png](https://github.com/grifguitar/metopt/blob/main/img/newplot8.png?raw=true)
![image.png](https://github.com/grifguitar/metopt/blob/main/img/newplot9.png?raw=true)
![image.png](https://github.com/grifguitar/metopt/blob/main/img/newplot10.png?raw=true)
![image.png](https://github.com/grifguitar/metopt/blob/main/img/newplot11.png?raw=true)
![image.png](https://github.com/grifguitar/metopt/blob/main/img/newplot12.png?raw=true)
![image.png](https://github.com/grifguitar/metopt/blob/main/img/newplot13.png?raw=true)
![image.png](https://github.com/grifguitar/metopt/blob/main/img/newplot14.png?raw=true)
![image.png](https://github.com/grifguitar/metopt/blob/main/img/newplot15.png?raw=true)

```haskell
! simple gradient, batch_size = all: 
Total time: 0.42996599999999996 milliseconds
Total iter: 8 iterations
Average time per iteration: 0.053745749999999995 milliseconds / iter

! minibatch gradient, batch_size = 32: 
Total time: 0.933943 milliseconds
Total iter: 7 iterations
Average time per iteration: 0.13342042857142858 milliseconds / iter

! stochastic gradient, batch_size = 1: 
Total time: 1.6718469999999999 milliseconds
Total iter: 65 iterations
Average time per iteration: 0.025720723076923076 milliseconds / iter

! norm simple gradient, batch_size = all: 
Total time: 0.49591599999999997 milliseconds
Total iter: 16 iterations
Average time per iteration: 0.030994749999999998 milliseconds / iter

! norm minibatch gradient, batch_size = 32: 
Total time: 0.9511569999999999 milliseconds
Total iter: 16 iterations
Average time per iteration: 0.059447312499999995 milliseconds / iter

! norm stochastic gradient, batch_size = 1: 
Total time: 3.33119 milliseconds
Total iter: 93 iterations
Average time per iteration: 0.03581924731182796 milliseconds / iter

! moment stochastic gradient, beta = 0.2: 
Total time: 0.7716449999999999 milliseconds
Total iter: 15 iterations
Average time per iteration: 0.051442999999999996 milliseconds / iter

! moment stochastic gradient, beta = 0.6: 
Total time: 4.486609 milliseconds
Total iter: 54 iterations
Average time per iteration: 0.08308535185185185 milliseconds / iter

! moment stochastic gradient, beta = 0.8: 
Total time: 0.997667 milliseconds
Total iter: 15 iterations
Average time per iteration: 0.06651113333333333 milliseconds / iter

! nesterov stochastic gradient, beta = 0.2: 
Total time: 1.831243 milliseconds
Total iter: 26 iterations
Average time per iteration: 0.07043242307692307 milliseconds / iter

! nesterov stochastic gradient, beta = 0.6: 
Total time: 1.745863 milliseconds
Total iter: 17 iterations
Average time per iteration: 0.10269782352941176 milliseconds / iter

! nesterov stochastic gradient, beta = 0.8: 
Total time: 1.08818 milliseconds
Total iter: 19 iterations
Average time per iteration: 0.05727263157894737 milliseconds / iter

! adagrad stochastic gradient: 
Total time: 3.2400889999999998 milliseconds
Total iter: 59 iterations
Average time per iteration: 0.05491676271186441 milliseconds / iter

! rmsprop stochastic gradient, beta = 0.2: 
Total time: 1.5607609999999998 milliseconds
Total iter: 19 iterations
Average time per iteration: 0.08214531578947368 milliseconds / iter

! rmsprop stochastic gradient, beta = 0.6: 
Total time: 1.21338 milliseconds
Total iter: 20 iterations
Average time per iteration: 0.060669 milliseconds / iter

! rmsprop stochastic gradient, beta = 0.8: 
Total time: 3.0592189999999997 milliseconds
Total iter: 56 iterations
Average time per iteration: 0.05462891071428572 milliseconds / iter
```
