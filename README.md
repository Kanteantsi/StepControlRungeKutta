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

Program compares establishing method, bisection method and 
Esposito-Based method of searching switching point in hybrid system, 
written as ODE

Program will end, when you try to use bisection for sqrt equation, or establishing
method. It happens, because you will cross the switching point, and will try to
get square root from negative value. It means, that this methods are not reliable.

Esposito method, in opposite, is reliable, and don't cross the switching point.