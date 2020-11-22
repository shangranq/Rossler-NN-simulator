# Rössler-NN-simulator
This repo contains systematic experiments of a set of neural network models to forecast the nonlinear dynamics of Rössler attractor.

# Introduction
### Rössler system 
Rössler system is a system of three non-linear ordinary differential equations originally studied by Otto Rössler. More details see (https://en.wikipedia.org/wiki/R%C3%B6ssler_attractor). The system of ODEs is described by the equations below:

<img src="plots/equation.png" width="200"/>

Given an initial condition, Runge-Kutta method was applied to find the numerical solution of the time series (see RosslerSolver.py for details). Then various nueral networks were explored to model the non-linear dynamics of Rössler attractors.    

# Visualization
For a special case where the parameters a=0.2, b=0.2, c=5.7 and the inital condition x=1, y=2, z=3, the trajectory of x(t), y(t), z(t) are shown below.
3D trajectory         |  xy projection         |  yz projection         |  xz projection
:--------------------:| :--------------------: | :--------------------: | :-------------------------:
![](plots/3D.jpg)     |  ![](plots/xy.jpg)     | ![](plots/yz.jpg)      | ![](plots/xz.jpg)  

Time series of x(t), y(t), z(t) are shown below as well. Even though the system is completely determinstic, small difference in the initial conditions can cause completely different trajectories in the far future. This phenomenon known as butterfly effect reflects the chaotic nature of the Rössler system. 

time series X         |  time series Y         |  time series Z         |  butterfly effect
:--------------------:| :--------------------: | :--------------------: | :-------------------------:
![](plots/time_x.jpg) |  ![](plots/time_y.jpg) | ![](plots/time_z.jpg)  | ![](plots/butterfly_x.jpg)  

# Modelling
