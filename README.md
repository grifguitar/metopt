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

! simple gradient, batch_size = all:
Total time: 0.23255299999999998 milliseconds Total iter: 8 iterations Average time per iteration: 0.029069124999999998
milliseconds / iter

! minibatch gradient, batch_size = 16:
Total time: 0.452017 milliseconds Total iter: 8 iterations Average time per iteration: 0.056502125 milliseconds / iter

! stochastic gradient, batch_size = 1:
Total time: 2.7630559999999997 milliseconds Total iter: 65 iterations Average time per iteration: 0.042508553846153846
milliseconds / iter

! norm simple gradient, batch_size = all:
Total time: 1.0764829999999999 milliseconds Total iter: 16 iterations Average time per iteration: 0.06728018749999999
milliseconds / iter

! norm minibatch gradient, batch_size = 16:
Total time: 0.570375 milliseconds Total iter: 15 iterations Average time per iteration: 0.038024999999999996
milliseconds / iter

! norm stochastic gradient, batch_size = 1:
Total time: 4.250802999999999 milliseconds Total iter: 93 iterations Average time per iteration: 0.04570755913978495
milliseconds / iter

! moment stochastic gradient, beta = 0.3:
Total time: 0.77069 milliseconds Total iter: 16 iterations Average time per iteration: 0.048168125 milliseconds / iter

! moment stochastic gradient, beta = 0.5:
Total time: 2.8243549999999997 milliseconds Total iter: 43 iterations Average time per iteration: 0.06568267441860465
milliseconds / iter

! moment stochastic gradient, beta = 0.8:
Total time: 0.655034 milliseconds Total iter: 15 iterations Average time per iteration: 0.04366893333333333 milliseconds
/ iter

! nesterov stochastic gradient, beta = 0.3:
Total time: 0.905532 milliseconds Total iter: 16 iterations Average time per iteration: 0.05659575 milliseconds / iter

! nesterov stochastic gradient, beta = 0.5:
Total time: 3.9669779999999997 milliseconds Total iter: 76 iterations Average time per iteration: 0.05219707894736842
milliseconds / iter

! nesterov stochastic gradient, beta = 0.8:
Total time: 0.629154 milliseconds Total iter: 19 iterations Average time per iteration: 0.033113368421052634
milliseconds / iter
