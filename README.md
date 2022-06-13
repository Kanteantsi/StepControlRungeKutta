# StepControlRungeKutta
Realisation of Runge-Kutta Integrating with event function step control

User must prepare Jacobian Matrix for searching step control, and write function
by pattern above, substituting needed math functions in sympy, otherwise, it wouldn't
work.

User determines the ODE system. Parameter b may be used for functions, based on time.
Method automatically controls step by accuracy. 
For controlling step by accuracy user can build function with Eiler Method as the
pattern above.

User determines the predicate function for ending integration.