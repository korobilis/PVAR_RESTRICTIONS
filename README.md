# PVAR_RESTRICTIONS
This code replicates the results in the paper  

Koop, G. and Korobilis, D. (2016).  Model Uncertainty in Panel Vector Autoregressive Models, European Economic Review 81, pp. 115-131.  

The code allows to search stochastically, and infer probabilistically (using our novel Stochastic Search Specification Selection, S4, algorithm), the existence of the following restrictions:  
1) Dynamic Interdependencies  
2) Cross-Sectional Heterogeneities  
3) Static Interdependencies  
in the context of panel VARs. 

One file estimates the model for Euro-Area data (see also the accompanying file for the Impulse responses), and the other implements our Monte Carlo exercise. 

There is also a small manual which clarifies the way we index inside the code the various restrictions in panel VARs.
