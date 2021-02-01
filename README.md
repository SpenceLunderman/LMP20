# LMP20
Bayesian Optimization of Bayesian Algorithms

The BOBA directory has two directories:

1) Estimation:
  - EnKF.py is the squareroot ensemble Kalman filter with ensemble inflation and covariance localization
  - GBO.py is the Global Bayesian Optimization algorithm with two aquisition functions: Expected Improvement and Knowledge Gradient
  - GP_kernel_search.py is a greedy kernel searching algorithm for Gaussian process kernels
  - PO_EnKF.py is the Perturbed Observation Ensemble Transform Kalman filter; no localization or inflation options
  - RBF_GP.py is the Radial Basis Function kernel for a Gaussian process; this is for testing purposes only
  
 2) Models:
  - Fuzzy_Quadratic.py is a n-dimentional quadratic function with a parameter to make it "fuzzy"; this is for testing purposes only
  - L63.py is a class with the Lorenz 1963 ODE, including the DA next step function M(x,theta).
  - L96.py is two Lorenz 1996 classes, each including the DA next step function M(x,theta). The two classes are L96_1 which has a state space X of nK dimension, all with the same scale. The other class is L96_23 which has two states X and Y, both have differing scales; there are nK X dimensions and each X state is coupled with  nJ Y dimensions.
  
  
RK.py is the Runge-Kutta 4th order method.
