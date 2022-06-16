import math
import numpy
import sympy
from sympy import zeros, Matrix, S
from functools import reduce
import time
from numpy import array, asarray, linspace, squeeze
from matplotlib.pyplot import figure, grid, plot, show, xlabel, ylabel, title, legend
from datetime import datetime

import matplotlib

x, y, z = sympy.symbols("x y z")

b2x2 = Matrix([[5.], [-3.]])

k = 3
g = 9.81


#hweilerbasedrungekutta = 1.39E-5


#Setting the predicate functions
def predicate_function_linear_xsecond(xstart,eps):
    return float(xstart[0]) < 0 + eps

#use if you need to stop integrate by time
def predicate_function_establ(xstart,eps):
    return False




J = sympy.Function('J')(x, y, z)


#Setting the Jacobian matrix elements
f1_swpx_angle_ball_step1 = 'arctg(x)'
print(f1_swpx_angle_ball_step1)
f1_swpy_angle_ball_step1 = 'y/(x*x+1)'
print(f1_swpx_angle_ball_step1)
f1_swpz_angle_ball_step1 = '0'
print(f1_swpx_angle_ball_step1)
f2_swpx_angle_ball_step1 = '-3*arctg(x)'
print(f2_swpx_angle_ball_step1)
f2_swpy_angle_ball_step1 = '-3*(x+3.27)/(y*y+1)'
print(f2_swpy_angle_ball_step1)
f2_swpz_angle_ball_step1 = '0'
print(f2_swpy_angle_ball_step1)
f3_swpx_angle_ball_step1 = "1/(x*x+1)"
print(f2_swpy_angle_ball_step1)
f3_swpy_angle_ball_step1 = "0"
print(f2_swpy_angle_ball_step1)
f3_swpz_angle_ball_step1 = "(0)"
print(f3_swpz_angle_ball_step1)

#Setting the Jacobian matrix elements
F3_switchball_angle_step1 = sympy.Matrix([[f1_swpx_angle_ball_step1, f1_swpy_angle_ball_step1, f1_swpz_angle_ball_step1],
                                    [f2_swpx_angle_ball_step1, f2_swpy_angle_ball_step1, f2_swpz_angle_ball_step1],
                                    [f3_swpx_angle_ball_step1, f3_swpy_angle_ball_step1, f3_swpz_angle_ball_step1]])


JacobianF_falling_angle_ball_step1 = sympy.Matrix(F3_switchball_angle_step1.subs([(x,5),(y,5),(z,0)]))

jaceiglist = list(JacobianF_falling_angle_ball_step1.eigenvals())

print(jaceiglist)
eigenlist = []


for i in range(len(jaceiglist)):
    eq = sympy.S(str(jaceiglist[i]))
    eq = eq.subs(sympy.Function('arctg'), sympy.atan)
    eq = eq.subs(sympy.Function('sqrt'), math.sqrt)
    eq = float((eq).evalf())
    eigenlist.append(eq)

hstability =  2 / (reduce(lambda x, y: abs(x) if abs(x) > abs(y) else abs(y), eigenlist))

print('Stability step for falling ball',hstability)



def rkf2stepcontrol(
        matrix_size,
        dydx, A, x, b, dxdt, hn, t,
        tout, eps, predicate_func,
        MatrixForYacobian, hstabilitygetting, halgebraic,esposito_on):

    iterations = 0
    endtime = 0
    with open(__file__ + 'OutputRungeKutta2.txt', 'w') as f:
        global needpredfunc
        start_time = datetime.now()

        #xbefprint = numpy.zeros(matrix_size, 1)
        xprint_s = numpy.zeros((matrix_size, 1))
        #newxfork2 = numpy.zeros(matrix_size, 1)

        k1 = numpy.zeros((matrix_size, 1))
        k2 = numpy.zeros((matrix_size, 1))
        k3 = numpy.zeros((matrix_size, 1))
        k2k1norm = numpy.zeros((matrix_size, 1))

        stability_criterion_array = []
        stability_criterion_array_numpy = numpy.zeros((matrix_size, 1))
        h_accuracy = 0
        h_stability = 0
        steps = [h_accuracy, h_stability, halgebraic]


        while not predicate_func(x, eps):
            if (esposito_on == False):
                if t - tout > eps:
                    break

            b = t
            iterations += 1
            print("I am inside ", x)

            xprint_s = numpy.c_[xprint_s, x]
            k1 = hn * dxdt(A, x, b)

            for i in (range(matrix_size)):
                k2[i, 0] = hn * dxdt(A, x + k1[i, 0] * numpy.ones(x.shape,dtype=float,order='C'), b)[i, 0]

            xold = x

            for i in (range(matrix_size)):
                x[i, 0] = x[i, 0] + 0.5 * (k1[i, 0] + k2[i, 0])

            t += hn

            hstability = 0
            hacc = 0
            h_accuracy = 0
            hbuffer = 0

            norm = float(0.5 * (k2 - k1).norm())

            # Next stage calculating

            if (0.5 * (k2 - k1).norm()) < eps and iterations >1:
                #hstability = hstabilitygetting(MatrixForYacobian, x)
                #print("New stability step", hstability)
                # MatrixForYacobian
                for i in range(matrix_size):
                    k3[i, 0] = hn * dxdt(A, x, b)[i, 0]

                for i in range(matrix_size):

                    stability_criterion_array_numpy[i,0] = abs(float(k3[i, 0] - k2[i, 0])) / abs(float(k2[i, 0] - k1[i, 0]))

                #'''
                if not (2*max(stability_criterion_array_numpy) <= 2):
                    hstability = hstabilitygetting(MatrixForYacobian, x)
                    steps[1] = hstability
                    #time.sleep(0.5)
                #'''


            elif (norm >= eps):  # Recalculate accuracy step
            #if (True):
                #h_stab = hstability

                hneeded = sympy.symbols('hn')

                j = 0

                for j in range(matrix_size):  # Automatic substitution

                    # k2k1norm[j, 0] = float(dxdt(dydx, A, x + k1[i, 0] * sympy.ones(*x.shape), b)[j, 0]) - float(dxdt(dydx, A, x, b)[j, 0])
                    k2k1norm[j, 0] = float(dxdt(A, x + k1[i, 0] * numpy.ones(x.shape,dtype=float,order='C'), b)[j, 0]) - \
                                     float(dxdt(A, x, b)[j, 0])

                epspart = str(2 * eps)

                equationstring = 'sqrt ('
                i = 0
                for i in range(matrix_size - 1):
                    equationstring = equationstring + '(hn*' + str(
                        k2k1norm[i,0]) + ')**2 +'  # automatic substitution
                    i += 1
                for i in range(matrix_size - 1, matrix_size):
                    equationstring = equationstring + '(hn*' + str(
                        k2k1norm[i,0]) + ')**2'
                    i += 1

                equationstring = equationstring + ') - ' + epspart
                #print(equationstring, 'Equation')
                hacc = sympy.solveset(equationstring, hneeded)

                if hacc.is_empty == True:
                    print("Accuracy step can't be found")
                    #time.sleep(3)
                elif (hacc.sup) <= 0:
                #    hbuffer = 0.000001
                    print("Accuracy step is zero or negative")
                    #time.sleep(3)

                h_accuracy = float(max(hacc))
                #print("Recalculating accuracy step", h_accuracy)
                steps[0] = h_accuracy
                #time.sleep(2)

            #print("Current steps",steps)
            #print("Steps", h_accuracy, hstability, halgebraic)
            hn = (min(n for n in steps  if n>0))
            print("Final current step", hn)

            print("Current time", t)
            endtime = t;

        end_time = datetime.now()
        print('Duration: {}'.format(end_time - start_time))

        return iterations, endtime, xprint_s


matrix_size = 3
k_angle_ball = 3
g = 9.81

A_Angleballbeforeeps = Matrix([k_angle_ball, g])
ystart_Angleballbeforeeps = Matrix([[5.], [5.],[0.]])  # xx,VV
dydx_Angleballbeforeeps = zeros(matrix_size, 1)

def dxdt_y_establstep1(A_Angleballbeforeeps, ystart_Angleballbeforeeps, b2x2):

    dydx_Angleballbeforeeps[0, 0] = ystart_Angleballbeforeeps[1, 0] * math.atan(ystart_Angleballbeforeeps[0, 0])
    dydx_Angleballbeforeeps[1, 0] = (-1 * A_Angleballbeforeeps[0, 0] * (ystart_Angleballbeforeeps[1, 0]) - 9.81)* math.atan(ystart_Angleballbeforeeps[0, 0])
    dydx_Angleballbeforeeps[2, 0] = math.atan(ystart_Angleballbeforeeps[0, 0])
    return dydx_Angleballbeforeeps


J1_switchball_angle_ball_step1 = sympy.Function('J')(x, y, z)

#setting Jacobian Matrix
f1_swpx_angle_ball_step1 = 'arctg(x)'
print(f1_swpx_angle_ball_step1)
f1_swpy_angle_ball_step1 = 'y/(x*x+1)'
print(f1_swpx_angle_ball_step1)
f1_swpz_angle_ball_step1 = '0'
print(f1_swpx_angle_ball_step1)
f2_swpx_angle_ball_step1 = '-3*arctg(x)'
print(f2_swpx_angle_ball_step1)
f2_swpy_angle_ball_step1 = '-3*(x+3.27)/(y*y+1)'
print(f2_swpy_angle_ball_step1)
f2_swpz_angle_ball_step1 = '0'
print(f2_swpy_angle_ball_step1)
f3_swpx_angle_ball_step1 = "1/(x*x+1)"
print(f2_swpy_angle_ball_step1)
f3_swpy_angle_ball_step1 = "0"
print(f2_swpy_angle_ball_step1)
f3_swpz_angle_ball_step1 = "(0)"
print(f3_swpz_angle_ball_step1)




F3_switchball_angle_step1 = Matrix([[f1_swpx_angle_ball_step1, f1_swpy_angle_ball_step1, f1_swpz_angle_ball_step1],
                                    [f2_swpx_angle_ball_step1, f2_swpy_angle_ball_step1, f2_swpz_angle_ball_step1],
                                    [f3_swpx_angle_ball_step1, f3_swpy_angle_ball_step1, f3_swpz_angle_ball_step1]])



print("Matrix for eigenvalues ",F3_switchball_angle_step1)

JacobianF_falling_angle_ball_step1 = Matrix(F3_switchball_angle_step1.subs([(x,5),(y,5),(z,0)]))

#eigenvalues
jaceiglist = list(JacobianF_falling_angle_ball_step1.eigenvals())

print(F3_switchball_angle_step1,"Jacobian Matrix")
print(F3_switchball_angle_step1.subs([(x,5),(y,5),(z,0)],"Substited values"))
print(F3_switchball_angle_step1.eigenvals(),"Eigenvalues")

#stability step function pattern
def hstabilitygetting_angle_ball_step1(MatrixForYacobian, values):

    JacobianF_falling_angle_ball_step1 = Matrix(
    MatrixForYacobian.subs([(x, values[0]), (y, values[1]), (z, values[2])]))
    jaceiglist = list(JacobianF_falling_angle_ball_step1.eigenvals())

    eigenlist = []

    for i in range(len(jaceiglist)):
        eq = sympy.S(str(jaceiglist[i]))
        eq = eq.subs(sympy.Function('arctg'), sympy.atan)
        eq = eq.subs(sympy.Function('sqrt'), math.sqrt)
        eq = float((eq).evalf())
        eigenlist.append(eq)

    return  2 / (reduce(lambda x, y: abs(x) if abs(x) > abs(y) else abs(y), eigenlist))



jaceiglist_falling_angle_ball_step1 = list(JacobianF_falling_angle_ball_step1.eigenvals())
hbegin_falling_ball_angle_ball_step1 = hstabilitygetting_angle_ball_step1(F3_switchball_angle_step1, [5,5,0])/400


print("Starting stability step for angle ball extended ODE", hbegin_falling_ball_angle_ball_step1)
time.sleep(5)

establish_step1_esposito_on = False
#'''##Need integration extended ODE falling ball
iterations_establ_angle_step1,endtime_establ_angle_step1, yprint_s_establ_angle_step1 = rkf2stepcontrol(matrix_size,
                                                                                    dydx_Angleballbeforeeps
                                                                                    , A_Angleballbeforeeps,
                                                                                    ystart_Angleballbeforeeps,
                                                                                    b2x2, dxdt_y_establstep1,
                                                                                    hbegin_falling_ball_angle_ball_step1,
                                                                                    # hbegin_falling_ball_step1_alt,
                                                                                    0
            #                                , 20, 0.1,predicate_function_establ_step1, F3, hstabilitygetting)
                                                                                    , 3.5,
                                                                                    0.000001,
                                                                                    predicate_function_establ,
                                                                                    F3_switchball_angle_step1,
                                                                                    hstabilitygetting_angle_ball_step1,
                                                                                    0.005,establish_step1_esposito_on)





#xprint_s_establ_angle_step1(0)
print("Shape of integrated matrix",yprint_s_establ_angle_step1.shape)
yprint_s_establ_angle_step1 = numpy.delete(yprint_s_establ_angle_step1, 0, -1)
print("Shape after deleting first column",yprint_s_establ_angle_step1.shape)

endtime_angle_ball_for_step2 = yprint_s_establ_angle_step1[2][-1]
print("Integration end time",endtime_angle_ball_for_step2)

#argument1 = list(reversed(yprint_s_establ_angle_step1[0]))
#argument2 = list(reversed(yprint_s_establ_angle_step1[1]))
#argument3 = list(reversed(yprint_s_establ_angle_step1[2]))


argument1 = list((yprint_s_establ_angle_step1[0]))
argument2 = list((yprint_s_establ_angle_step1[1]))
argument3 = list((yprint_s_establ_angle_step1[2]))

print('Amount of iterations',iterations_establ_angle_step1)
print('Время окончания', endtime_establ_angle_step1)

t = linspace(0, float(endtime_establ_angle_step1), yprint_s_establ_angle_step1.shape[1])

title('Establishing method falling ball step 1 Vy"=-k*Vy-g')
plot(t, argument1, '-o', linewidth=2)
plot(t, argument2, '-o', linewidth=2)
plot(t, argument3, '-o', linewidth=2)
legend(["y", "Vy", "t"], loc ="upper right")
print()
ylabel("argument")
xlabel("t")
grid(True)
show() # display
#'''

#Establishing Method for original
#endtime_angle_ball_for_step2 = 2.37349620145370
matrix_size = 2
k_angle_ball = 3
g = 9.81
A_Angleballaftereps = Matrix([k_angle_ball, g])
dydx_Angleballaftereps = zeros(matrix_size, 1)

def dxdt_y_establstep2(A_Angleballaftereps, ystart_Angleballaftereps, b2x2):


    dydx_Angleballaftereps[0, 0] = ystart_Angleballaftereps[1, 0]
    dydx_Angleballaftereps[1, 0] = (-1 * A_Angleballaftereps[0, 0] * (ystart_Angleballaftereps[1, 0]) - 9.81)  ##Vy' = -k*Vy-g

    return dydx_Angleballaftereps



J1_switchball_angle_ball_step2 = sympy.Function('J')(x, y)

#setting Jacobian Matrix
f1_swpx_angle_ball_step2 = '0'
f1_swpy_angle_ball_step2 = '1'
f2_swpx_angle_ball_step2 = '0'
f2_swpy_angle_ball_step2 = '3'

F3_switchball_angle_step2 = Matrix([[f1_swpx_angle_ball_step2, f1_swpy_angle_ball_step2],
                                    [f2_swpx_angle_ball_step2, f2_swpy_angle_ball_step2]])

JacobianF_falling_angle_ball_step2 = Matrix(F3_switchball_angle_step1.subs([(x,5),(y,5)]))
print('Jacobian Matrix for original ODE', JacobianF_falling_angle_ball_step2)
jaceiglist = list(JacobianF_falling_angle_ball_step2.eigenvals())
print('Eigenvalues original ODE', jaceiglist)

#stability function for original ODE
def hstabilitygetting_angle_ball_step2(MatrixForYacobian, values):
    JacobianF_falling_angle_ball_step2 = Matrix(
    MatrixForYacobian.subs([(x, values[0]), (y, values[1])]))
    jaceiglist = list(JacobianF_falling_angle_ball_step2.eigenvals())

    eigenlist = []

    for i in range(len(jaceiglist)):
        eq = sympy.S(str(jaceiglist[i]))
        eq = eq.subs(sympy.Function('arctg'), sympy.atan)
        eq = eq.subs(sympy.Function('sqrt'), math.sqrt)
        eq = float((eq).evalf())
        eigenlist.append(eq)

    return  2 / (reduce(lambda x, y: abs(x) if abs(x) > abs(y) else abs(y), eigenlist))



jaceiglist_falling_angle_ball_step2 = list(JacobianF_falling_angle_ball_step2.eigenvals())


hbegin_falling_ball_angle_ball_step2 = hstabilitygetting_angle_ball_step2(F3_switchball_angle_step2, [5.,5.])/4000
print("Beginning step or original ODE by stability", hbegin_falling_ball_angle_ball_step2)
ystart_Angleballaftereps = Matrix([[5.], [5.]])  # xx,VV


establish_step2_esposito_on = False
#'''##Need integration original ODE angle ball
iterations_establ_angle_step2,endtime_establ_angle_step2, yprint_s_establ_angle_step2 = rkf2stepcontrol(matrix_size,
                                                                                    dydx_Angleballaftereps
                                                                                    , A_Angleballaftereps,
                                                                                    ystart_Angleballaftereps,
                                                                                    b2x2, dxdt_y_establstep2,
                                                                                    hbegin_falling_ball_angle_ball_step2,
                                                                                    0
                                                                                    , endtime_angle_ball_for_step2,
                                                                                    #time from extended ODE
                                                                                    0.0000001,
                                                                                    predicate_function_establ,
                                                                                    F3_switchball_angle_step2,
                                                                                    hstabilitygetting_angle_ball_step2,
                                                                                    0.001,establish_step2_esposito_on)





yprint_s_establ_angle_step2 = numpy.delete(yprint_s_establ_angle_step2, 0, -1)
argument1 = list((yprint_s_establ_angle_step2[0]))
argument2 = list((yprint_s_establ_angle_step2[1]))

print('Amount of iterations',iterations_establ_angle_step2)
print('Time of end', endtime_establ_angle_step2)

t = linspace(0, float(endtime_establ_angle_step2), yprint_s_establ_angle_step2.shape[1])


title('Establishing method falling ball step 2 Vy"=-k*Vy-g')
plot(t, argument1, '-o', linewidth=2)
plot(t, argument2, '-o', linewidth=2)
legend(["y", "Vy", "t"], loc ="upper right")
print()
ylabel("argument")
xlabel("t")
grid(True)
show() # display
#'''

ystartlinearelersecondvarstep = Matrix([5, 5.,5.])  # x',Vx'
heilersecond_angle_ball = 2 / 1000  # 2.E-3
hpeilersecondvarstep_angle_ball = 0
hweilersecondvarstep_angle_ball = 0

gammaeilersecond_angle_ball = 0.1
epsilon_eiler_second_angle_ball = 0.0001

N_eiler_linear_second = 70


hw_pivot_angle_ball = sympy.zeros(1,1)

print("Parameters",A_Angleballaftereps,
                                                                                          ystartlinearelersecondvarstep,
                                                                                          gammaeilersecond_angle_ball,
                                                                                          hweilersecondvarstep_angle_ball,
                                                                                          hpeilersecondvarstep_angle_ball,
                                                                                          heilersecond_angle_ball,
                                                                                          epsilon_eiler_second_angle_ball,
                                                                                          N_eiler_linear_second, hw_pivot_angle_ball)


def eilermethod_angle_ball(Alineareilersecond, ystartlinearelersecondvarstep, gammaeilersecond_angle_ball,
                            hweilervarstep_angle_ball, hpeilersecondvarstep_angle_ball, heilersecond_angle_ball, epsilon_eiler_second_angle_ball,
                            N_eiler_linear_second,hweeee):

    start_time = datetime.now()
    print("Inside eiler angle ball")
    Yold = ystartlinearelersecondvarstep[0, 0]  # xold = x
    VYold = ystartlinearelersecondvarstep[1, 0]  # vxold = vx
    endtime_eilermethodvariablestep = 0

    h_for_output = []
    event_vect = []
    ystartlineareler_discrete_print_variablstep = zeros(3, 1)
    ##and (hweilervarstep_angle_ball > 0)
    while (ystartlinearelersecondvarstep[2, 0] > epsilon_eiler_second_angle_ball):  # g>eps
        #print("Inside Eiler angle ball")
        #print("Current time", endtime_eilermethodvariablestep)
        ystartlinearelersecondvarstep[2, 0] = ystartlinearelersecondvarstep[0, 0]  # g = x

        hpeilersecondvarstep_angle_ball = (gammaeilersecond_angle_ball - 1) * ystartlinearelersecondvarstep[2, 0] \
                               / (-3 * VYold - 9.81 + VYold)  # hp = (gamma-1)*g/(-omega**2*xold+Vxold)
        #There must be event control function  (gamma-1)*diffentials of needed original ODE



        event_vector = ystartlinearelersecondvarstep[2, 0] / (-3 * VYold - 9.81 + VYold)
        #print("Hp by event func  and h input by user",hpeilersecondvarstep_angle_ball, heilersecond_angle_ball)
        hweilersecondvarstep_angle_ball = min(hpeilersecondvarstep_angle_ball, heilersecond_angle_ball)  # hw = hp
        #print("Step after", hweilersecondvarstep_angle_ball)
        endtime_eilermethodvariablestep = endtime_eilermethodvariablestep + hweilersecondvarstep_angle_ball

        Fi = Matrix([VYold, -3  * VYold - 9.81])
        ystartlinearelersecondvarstep[0, 0] = Yold + hweilersecondvarstep_angle_ball * Fi[0, 0]  # x = xold+hw*Fi[1]
        ystartlinearelersecondvarstep[1, 0] = VYold + hweilersecondvarstep_angle_ball * Fi[1, 0]  # vx = vxold+hw*Fi[2]
        Yold = ystartlinearelersecondvarstep[0, 0]  # xold = x
        VYold = ystartlinearelersecondvarstep[1, 0]  # vxold = vx
        ystartlineareler_discrete_print_variablstep = \
            ystartlineareler_discrete_print_variablstep.col_insert(1, Matrix([ystartlinearelersecondvarstep]))
        #print("Integrated parameter", ystartlinearelersecondvarstep)

        h_for_output.append(hweilersecondvarstep_angle_ball)
        event_vect.append(event_vector)
        hweeee = hweeee.col_insert(1, Matrix([hweilersecondvarstep_angle_ball]))
        #print("Step vector",hweeee)
        if (ystartlinearelersecondvarstep[2, 0] <= epsilon_eiler_second_angle_ball):  # if g<=eps

            end_time = datetime.now()
            print('Duration: {}'.format(end_time - start_time))
            return ystartlineareler_discrete_print_variablstep, endtime_eilermethodvariablestep,h_for_output,event_vect




#'''#Eiler method call
solutionlineareiler_angle_ball, endtime_eiler_angle_ball,hw_pivot_angle_ball,event_vect = eilermethod_angle_ball(A_Angleballaftereps,
                                                                                          ystartlinearelersecondvarstep,
                                                                                          gammaeilersecond_angle_ball,
                                                                                          hweilersecondvarstep_angle_ball,
                                                                                          hpeilersecondvarstep_angle_ball,
                                                                                          heilersecond_angle_ball,
                                                                                          epsilon_eiler_second_angle_ball,
                                                                                          N_eiler_linear_second, hw_pivot_angle_ball)

solutionlineareiler_angle_ball.col_del(0)

argument1 = list(reversed(solutionlineareiler_angle_ball.row(0)))
argument2 = list(reversed(solutionlineareiler_angle_ball.row(1)))
argument3 = list(reversed(solutionlineareiler_angle_ball.row(2)))
argument4 = list(reversed(event_vect))

print('End time for eiler', endtime_eiler_angle_ball)

t = linspace(0, float(endtime_eiler_angle_ball), solutionlineareiler_angle_ball.shape[1])

title('Runge-Kutta Novikov-Shornikov Eiler Falling Ball ')
plot(t, argument1, '-o', linewidth=2)
plot(t, argument2, '-o', linewidth=2)
legend(["y", "Vy"], loc ="lower left")
print()
ylabel("argument")
xlabel("t")
grid(True)
show() # display

title('Esposito step control based on Newton"s method Falling Ball ')
plot(t, argument3, '-o', linewidth=2)
legend(["h"], loc ="lower left")
print()
ylabel("argument")
xlabel("t")
grid(True)
show() # display

title('Event function value Falling Ball ')
plot(t, argument4, '-o', linewidth=2)
legend(["value"], loc ="lower left")
print()
ylabel("argument")
xlabel("t")
grid(True)
show() # display

hweilerbasedrungekutta = hw_pivot_angle_ball[-1]
print("Eiler method based Step", hweilerbasedrungekutta)

#'''#Eiler method

#hweilerbasedrungekutta = 1.39E-5

def predicate_function_linear_xsecond(xstart,eps):

    return float(xstart[0]) < 0 + eps

rkeiler_esposito_on = True

A_Angleball_esposito = Matrix([k_angle_ball, g])
ystart_Angleball_esposito = Matrix([[5.], [5.]])
dydx_Angleball_esposito = zeros(matrix_size, 1)

#'''# Esposito method step control by event function, step got from Eiler method
iterations_rkeiler_angle_step2,endtime_rkeiler_angle_step2, yprint_s_rkeiler_angle_step2 = rkf2stepcontrol(matrix_size,
                                                                                    dydx_Angleball_esposito
                                                                                    , A_Angleball_esposito,
                                                                                    ystart_Angleball_esposito,
                                                                                    b2x2, dxdt_y_establstep2,
                                                                                       hweilerbasedrungekutta,
                                                                                    0
                                                                                    , 10,
                                                                                    epsilon_eiler_second_angle_ball,
                                                                                    #needed accuracy
                                                                                    predicate_function_linear_xsecond,
                                                                                    F3_switchball_angle_step2,
                                                                                    hstabilitygetting_angle_ball_step2,
                                                                                    hweilerbasedrungekutta,
                                                                                    rkeiler_esposito_on)


yprint_s_rkeiler_angle_step2 = numpy.delete(yprint_s_rkeiler_angle_step2, 0, -1)
argument1 = list((yprint_s_rkeiler_angle_step2[0]))
argument2 = list((yprint_s_rkeiler_angle_step2[1]))


print('Amount of iterations',iterations_rkeiler_angle_step2)
print('End of integration', endtime_rkeiler_angle_step2)
t = linspace(0, float(endtime_rkeiler_angle_step2), yprint_s_rkeiler_angle_step2.shape[1])

title('Runge Kutta Esposito Based on Eiler')
plot(t, argument1, '-o', linewidth=2)
plot(t, argument2, '-o', linewidth=2)
legend(["y", "Vy"], loc ="upper right")
print()
ylabel("argument")
xlabel("t")
grid(True)
show() # display
#'''#Esposito method end



#Bisection part start

hn = 0.001

Aklinnonlin = Matrix([[k, 1], [0, -0.5]])

dydxbisect = zeros(matrix_size, 1)

print(dydxbisect)

xstartswitchpoint = Matrix([[5.], [5.]])


def dxdt_y(Aklinnonlin, xstartswitchpoint, b2x2):

    dydxbisect[0, 0] = xstartswitchpoint[1, 0]
    dydxbisect[1, 0] = -1 * Aklinnonlin[0, 0] * (xstartswitchpoint[1, 0]) - 9.81  ##Vy' = -k*Vy-g
    return dydxbisect


k1 = zeros(matrix_size, 1)
k2 = zeros(matrix_size, 1)
k3 = zeros(matrix_size, 1)
k2k1norm = zeros(matrix_size, 1)


xprint_s_bisect2 = Matrix([5.,5.])
xstartswitchpoint2 = Matrix([5.,5.])
iterations2 = 0
matrix_size = 2
t2 = 0

def predicate_function_linear_xsecond(xstart,eps):
    return float(xstart[0]) < 0 + eps



start_time = datetime.now()
while True:  # Integrate
    for i in (range(matrix_size)):
        iterations2 += 1
        k1[i, 0] = hn * dxdt_y(Aklinnonlin, xstartswitchpoint2, b2x2)[i, 0]
        k2[i, 0] = hn * dxdt_y(
            Aklinnonlin,
            xstartswitchpoint2 + k1[i, 0] * sympy.ones(*xstartswitchpoint2.shape),
            b2x2,
        )[i, 0]
        xstartswitchpoint2[i, 0] = xstartswitchpoint2[i, 0] + 0.5 * (k1[i, 0] + k2[i, 0])
        xprint_s_bisect2 = xprint_s_bisect2.col_insert(1, Matrix([xstartswitchpoint2]))
    dxdt_y(Aklinnonlin, xstartswitchpoint2, b2x2)
    print("Integrated parameter, Runge-Kutta",xstartswitchpoint2)
    print(t2)

    t2 += hn
    if predicate_function_linear_xsecond(xstartswitchpoint2,epsilon_eiler_second_angle_ball/1000):
        '''
        iterations += 1
        xprint_s_bisect = xprint_s_bisect.col_insert(1, Matrix([xstartswitchpoint]))
        for i, number in enumerate(range(matrix_size)):
            k1[i, 0] = hn * dxdt_y(Aklinnonlin, xstartswitchpoint, b2x2)[i, 0]
            k2[i, 0] = hn * dxdt_y(
                Aklinnonlin,
                xstartswitchpoint + k1[i, 0] * sympy.ones(*xstartswitchpoint.shape),
                b2x2,
            )[i, 0]
            xstartswitchpoint[i, 0] = xstartswitchpoint[i, 0] + 0.5 * (k1[i, 0] + k2[i, 0])
        dxdt_y(Aklinnonlin, xstartswitchpoint, b2x2)
        t += hn
        '''
        break
    end_time = datetime.now()
    print('Duration Bisection: {}'.format(end_time - start_time))


xprint_s_bisect2.col_del(0)  # Delete first column

argument1 = list(reversed(xprint_s_bisect2.row(0)))
argument2 = list(reversed(xprint_s_bisect2.row(1)))

t = linspace(0, t2, iterations2)



title('Bisection falling ball')
plot(t, argument1, '-o', linewidth=2)
plot(t, argument2, '-o', linewidth=2)
legend(["y", "Vy"], loc ="upper right")
print()
ylabel("argument")
xlabel("t")
grid(True)
show() # display

'''     #Bisection localization

print('End time', t)
eps = epsilon_eiler_second_angle_ball
tmax = t
tmin = t - hn
hnew = (tmax - tmin) / 2
tmiddle = tmin + hnew

xleft = xprint_s_bisect2.col(1)   #change to numpy pattern
xright = xprint_s_bisect2.col(0)  #change to numpy pattern

xmiddle = zeros(matrix_size, 1)

# print("X слева",xleft)                  
# print("X справа",xright)                 
# print("T слева",tmin)
# print("T посередине",tmiddle)
# print("T справа",tmax)

def bisectionrk(xleft, xmiddle, xright, hnew, tleft, tmiddle, tright, eps):
    while (tmiddle - tleft) > eps or (tright - tmiddle) > eps:
        # while (tmiddle - tleft) >= eps:

        # print()
        # print(f"Before values {'Слева ', xleft[0, 0]} {'Посередине ', xmiddle[0, 0]} {'Справа ', xright[0, 0]}")
        # print(
        #    f"Before values {'Слева ', predfunc(xleft)} {'Посередине ', predfunc(xmiddle)} {'Справа ', predfunc(xright)}")
        # print()

        for i in range(matrix_size):
            k1[i, 0] = hnew * dxdt_y(Aklinnonlin, xleft, b2x2)[i, 0]
            k2[i, 0] = (
                    hnew
                    * dxdt_y(
                Aklinnonlin, xleft + k1[i, 0] * sympy.ones(*xleft.shape), b2x2
            )[i, 0]
            )
            xmiddle[i, 0] = xleft[i, 0] + 0.5 * (k1[i, 0] + k2[i, 0])
        dxdt_y(Aklinnonlin, xmiddle, b2x2)

        if predfunc(xmiddle) == True:
            if predfunc(xleft) == False and (tmiddle - tleft) <= eps:
                return xmiddle
            elif ((tmiddle - tleft) > eps):
                hnew = (tmiddle - tleft) / 2  
                tright = tmiddle  
                tmiddle = (
                        tleft + hnew
                ) 
                xright = xmiddle 
                bisectionrk(
                    xleft, xmiddle, xright, hnew, tleft, tmiddle, tright, eps
                )
            elif (predfunc(xleft) == True):
                print(Localization is impossible")
                break;

        elif predfunc(xmiddle) == False:
            if predfunc(xright) == True and (tright - tmiddle) <= eps:
                return xright
            elif (tright - tmiddle) > eps:
                hnew = (tright - tmiddle) / 2 
                tleft = tmiddle  
                tmiddle = tmiddle + hnew  
                xleft = xmiddle  
                bisectionrk(
                    xleft, xmiddle, xright, hnew, tleft, tmiddle, tright, eps
                )
            elif (predfunc(xright) == False):
                # print("I am 4")
                print("Localization is impossible")
                break;

    if (tmiddle - tleft) < eps and predfunc(xmiddle):
        return xmiddle, tmiddle
    elif (tright - tmiddle) < eps and predfunc(xright):
        return xright, tright

startingpointx,switchtime = bisectionrk(xleft, xmiddle, xright, hnew, tmin, tmiddle, tmax, eps)
print("Final localization point, x = ", startingpointx, 't = ',switchtime)


'''#Bisection part end

ystartlinearelersecondvarstepsqrt = Matrix([5, 5.,5.])  # x',Vx'
heilersecond_angle_ball_sqrt = 2 / 1000  # 2.E-3
hpeilersecondvarstep_angle_sqrt = 0
hweilersecondvarstep_angle_sqrt = 0

gammaeilersecond_angle_sqrt = 0.1
epsilon_eiler_sqrt = 0.0001

N_eiler_linear_secondsqrt = 70


hw_pivot_angle_ballsqrt = sympy.zeros(1,1)

print("Parameters",A_Angleballaftereps,
                                                                                          ystartlinearelersecondvarstepsqrt,
                                                                                          gammaeilersecond_angle_sqrt,
                                                                                          hweilersecondvarstep_angle_sqrt,
                                                                                          hpeilersecondvarstep_angle_sqrt,
                                                                                          heilersecond_angle_ball_sqrt,
                                                                                          epsilon_eiler_sqrt,
                                                                                          N_eiler_linear_secondsqrt,
                                                                                        hw_pivot_angle_ballsqrt)


def eilermethod_angle_ball_sqrt(Alineareilersecond, ystartlinearelersecondvarstep_sqrt, gammaeilersecond_angle_ball_sqrt,
                            hweilervarstep_angle_sqrt, hpeilersecondvarstep_angle_sqrt, heilersecond_angle_ball_sqrt, epsilon_eiler_second_angle_sqrt,
                            N_eiler_linear_second_sqrt,hweeee):

    start_time = datetime.now()
    print("Inside eiler angle ball")
    Yold_sqrt = ystartlinearelersecondvarstep_sqrt[0, 0]  # xold = x
    VYold_sqrt = ystartlinearelersecondvarstep_sqrt[1, 0]  # vxold = vx
    endtime_eilermethodvariablestep_sqrt = 0

    h_for_output_sqrt = []
    event_vect_sqrt = []
    ystartlineareler_discrete_print_variablstep_sqrt = zeros(3, 1)
    ##and (hweilervarstep_angle_ball > 0)
    while (ystartlinearelersecondvarstep_sqrt[2, 0] > epsilon_eiler_second_angle_sqrt):  # g>eps
        #print("Inside Eiler angle ball")
        #print("Current time", endtime_eilermethodvariablestep)
        ystartlinearelersecondvarstep_sqrt[2, 0] = ystartlinearelersecondvarstep_sqrt[0, 0]  # g = x

        hpeilersecondvarstep_angle_sqrt = (gammaeilersecond_angle_ball_sqrt - 1) * ystartlinearelersecondvarstep_sqrt[2, 0] \
                               / (-25*math.sqrt(ystartlinearelersecondvarstep_sqrt[0, 0])+ystartlinearelersecondvarstep_sqrt[1,0])  # hp = (gamma-1)*g/(-omega**2*xold+Vxold)
        #There must be event control function  (gamma-1)*diffentials of needed original ODE



        event_vector_sqrt = ystartlinearelersecondvarstep_sqrt[2, 0] /  (ystartlinearelersecondvarstep_sqrt[1, 0]-25*math.sqrt((ystartlinearelersecondvarstep_sqrt[0,0])))
        #print("Hp by event func  and h input by user",hpeilersecondvarstep_angle_ball, heilersecond_angle_ball)
        hweilervarstep_angle_sqrt = min(hpeilersecondvarstep_angle_sqrt, heilersecond_angle_ball_sqrt)  # hw = hp
        print("Step after", hweilervarstep_angle_sqrt)
        endtime_eilermethodvariablestep_sqrt = endtime_eilermethodvariablestep_sqrt + hweilervarstep_angle_sqrt

        Fi_sqrt = Matrix([VYold_sqrt, -25*math.sqrt(Yold_sqrt)])
        ystartlinearelersecondvarstep_sqrt[0, 0] = Yold_sqrt + hweilervarstep_angle_sqrt * Fi_sqrt[0, 0]  # x = xold+hw*Fi[1]
        ystartlinearelersecondvarstep_sqrt[1, 0] = VYold_sqrt + hweilervarstep_angle_sqrt * Fi_sqrt[1, 0]  # vx = vxold+hw*Fi[2]
        Yold_sqrt = ystartlinearelersecondvarstep_sqrt[0, 0]  # xold = x
        VYold_sqrt = ystartlinearelersecondvarstep_sqrt[1, 0]  # vxold = vx
        ystartlineareler_discrete_print_variablstep_sqrt = \
            ystartlineareler_discrete_print_variablstep_sqrt.col_insert(1, Matrix([ystartlinearelersecondvarstep_sqrt]))
        print("Integrated parameter", ystartlinearelersecondvarstep)

        h_for_output_sqrt.append(hweilervarstep_angle_sqrt)
        event_vect_sqrt.append(event_vector_sqrt)
        hweeee = hweeee.col_insert(1, Matrix([hweilervarstep_angle_sqrt]))
        #print("Step vector",hweeee)
        if (ystartlinearelersecondvarstep_sqrt[2, 0] <= epsilon_eiler_second_angle_sqrt):  # if g<=eps

            end_time = datetime.now()
            print('Duration: {}'.format(end_time - start_time))
            return ystartlineareler_discrete_print_variablstep_sqrt, endtime_eilermethodvariablestep_sqrt,h_for_output_sqrt,event_vect_sqrt


A_Angleball_sqrt = Matrix([k_angle_ball, g])
ystart_Angleball_sqrt = Matrix([[5.], [5.],[5.]])

#'''#Eiler method call
solutionlineareiler_angle_sqrt, endtime_eiler_angle_sqrt,hw_pivot_angle_sqrt,event_vect_sqrt = eilermethod_angle_ball_sqrt(A_Angleball_sqrt,
                                                                                          ystart_Angleball_sqrt,
                                                                                          gammaeilersecond_angle_sqrt,
                                                                                          hweilersecondvarstep_angle_sqrt,
                                                                                          hpeilersecondvarstep_angle_sqrt,
                                                                                          heilersecond_angle_ball_sqrt,
                                                                                          epsilon_eiler_sqrt,
                                                                                          N_eiler_linear_secondsqrt,
                                                                                        hw_pivot_angle_ballsqrt)

solutionlineareiler_angle_sqrt.col_del(0)

argument1 = list(reversed(solutionlineareiler_angle_sqrt.row(0)))
argument2 = list(reversed(solutionlineareiler_angle_sqrt.row(1)))
argument3 = list(reversed(solutionlineareiler_angle_sqrt.row(2)))
argument4 = list(reversed(event_vect_sqrt))

print('Time of end for Eiler sqrt', endtime_eiler_angle_sqrt)

t = linspace(0, float(endtime_eiler_angle_sqrt), solutionlineareiler_angle_sqrt.shape[1])

title('Eiler Falling Ball Sqrt ')
plot(t, argument1, '-o', linewidth=2)
plot(t, argument2, '-o', linewidth=2)
legend(["y", "Vy"], loc ="lower left")
print()
ylabel("argument")
xlabel("t")
grid(True)
show() # display

title('Esposito step control based on Newton"s method Falling Ball Sqrt')
plot(t, argument3, '-o', linewidth=2)
legend(["h"], loc ="lower left")
print()
ylabel("argument")
xlabel("t")
grid(True)
show() # display

title('Event function value Falling Ball sqrt ')
plot(t, argument4, '-o', linewidth=2)
legend(["value"], loc ="lower left")
print()
ylabel("argument")
xlabel("t")
grid(True)
show() # display

hweilerbasedrungekutta_sqrt = hw_pivot_angle_sqrt[-1]
print("Step,based on Eiler, with sqrt°", hweilerbasedrungekutta_sqrt)


#'''#Eiler method SQRT

#hweilerbasedrungekutta = 1.39E-5

def predicate_function_linear_xsecond(xstart,eps):

    return float(xstart[0]) < 0 + eps

rkeiler_esposito_on = True

A_Angleball_esposito_sqrt = Matrix([k_angle_ball, g])  # СЃРјРµРЅР° Numpy РјР°С‚СЂРёС†С‹ РЅР° Sympy
ystart_Angleball_esposito_sqrt = Matrix([[5.], [5.]])  # xx,VV            #СЃРјРµРЅР° Numpy РјР°С‚СЂРёС†С‹ РЅР° Sympy
dydx_Angleball_esposito_sqrt = zeros(matrix_size, 1)


def dxdt_y_sqrt(A_Angleball_esposito_sqrt, ystart_Angleball_esposito_sqrt, b2x2):  # СЃРёСЃС‚РµРјР° СѓСЂР°РІРЅРµРЅРёР№ РґР»СЏ РїСЂРѕРІРµСЂРєРё РїРѕРёСЃРєР° С‚РѕС‡РєРё РїРµСЂРµРєР»СЋС‡РµРЅРёСЏ
    dydx_Angleball_esposito_sqrt[0, 0] = ystart_Angleball_esposito_sqrt[1, 0]
    dydx_Angleball_esposito_sqrt[1, 0] = -25*math.sqrt((ystart_Angleball_esposito_sqrt[0, 0]))  ##Vy' = -k*Vy-g

    return dydx_Angleball_esposito_sqrt


#'''# Esposito method step control by event function SQRT, step got from Eiler method
iterations_rkeiler_angle_sqrt,endtime_rkeiler_angle_sqrt, yprint_s_rkeiler_angle_sqrt = rkf2stepcontrol(matrix_size,
                                                                                    dydx_Angleball_esposito_sqrt
                                                                                    , A_Angleball_esposito_sqrt,
                                                                                    ystart_Angleball_esposito_sqrt,
                                                                                    b2x2, dxdt_y_sqrt,
                                                                                       hweilerbasedrungekutta_sqrt,
                                                                                    0
                                                                                    , 10,
                                                                                    epsilon_eiler_sqrt,
                                                                                    #needed accuracy
                                                                                    predicate_function_linear_xsecond,
                                                                                    F3_switchball_angle_step2,
                                                                                    hstabilitygetting_angle_ball_step2,
                                                                                    hweilerbasedrungekutta_sqrt,
                                                                                    rkeiler_esposito_on)


yprint_s_rkeiler_angle_sqrt = numpy.delete(yprint_s_rkeiler_angle_sqrt, 0, -1)
argument1 = list((yprint_s_rkeiler_angle_sqrt[0]))
argument2 = list((yprint_s_rkeiler_angle_sqrt[1]))


print('Amount of iterations',iterations_rkeiler_angle_sqrt)
print('End of integration', endtime_rkeiler_angle_sqrt)
t = linspace(0, float(endtime_rkeiler_angle_sqrt), yprint_s_rkeiler_angle_sqrt.shape[1])

title('Runge Kutta Esposito Based on Eiler sqrt')
plot(t, argument1, '-o', linewidth=2)
plot(t, argument2, '-o', linewidth=2)
legend(["y", "Vy"], loc ="upper right")
print()
ylabel("argument")
xlabel("t")
grid(True)
show() # display
#'''#Esposito method end

#'''#Establishing Method sqrt step1 begin
matrix_size_sqrt_est1 = 3
k_angle_ball = 3
g = 9.81

A_Angleball_sqrt_est1 = Matrix([k_angle_ball, g]) # смена Numpy матрицы на Sympy
ystart_Angleball_sqrt_est1 = Matrix([[5.], [5.],[0.]]) # xx,VV #смена Numpy матрицы на Sympy
dydx_Angleball_sqrt_est1 = zeros(matrix_size_sqrt_est1, 1)

def dxdt_y_establ_sqrt_est1(A_Angleball_sqrt_est1, ystart_Angleball_sqrt_est1, b2x2): # система уравнений для проверки поиска точки переключения

    dydx_Angleball_sqrt_est1[0, 0] = ystart_Angleball_sqrt_est1[1, 0] * math.atan(ystart_Angleball_sqrt_est1[0, 0])
    dydx_Angleball_sqrt_est1[1, 0] = (-25*math.sqrt(ystart_Angleball_sqrt_est1[0,0]))* math.atan(ystart_Angleball_sqrt_est1[0, 0])
    dydx_Angleball_sqrt_est1[2, 0] = math.atan(ystart_Angleball_sqrt_est1[0, 0])
    return dydx_Angleball_sqrt_est1

J1_switchball_angle_ball_step1 = sympy.Function('J')(x, y, z)

#setting Jacobian Matrix
f1_swpx_angle_ball_sqrt_est1 = 'y/(x*x+1)'
print(f1_swpx_angle_ball_step1)
f1_swpy_angle_ball_sqrt_est1 = 'arctg(x)'
print(f1_swpx_angle_ball_step1)
f1_swpz_angle_ball_sqrt_est1 = '0'
print(f1_swpx_angle_ball_step1)
f2_swpx_angle_ball_sqrt_est1 = '-25*((arctg(x))/(2*sqrt(x))+(sqrt(x))/(x*x+1))'
print(f2_swpx_angle_ball_step1)
f2_swpy_angle_ball_sqrt_est1 = '0'
print(f2_swpy_angle_ball_step1)
f2_swpz_angle_ball_sqrt_est1 = '0'
print(f2_swpy_angle_ball_step1)
f3_swpx_angle_ball_sqrt_est1 = "1/(x*x+1)"
print(f2_swpy_angle_ball_step1)
f3_swpy_angle_ball_sqrt_est1 = "0"
print(f2_swpy_angle_ball_step1)
f3_swpz_angle_ball_sqrt_est1 = "0"
print(f3_swpz_angle_ball_step1)

F3_switchball_angle_sqrt_est1 = Matrix([[f1_swpx_angle_ball_sqrt_est1, f1_swpy_angle_ball_sqrt_est1, f1_swpz_angle_ball_sqrt_est1],
[f2_swpx_angle_ball_sqrt_est1, f2_swpy_angle_ball_sqrt_est1, f2_swpz_angle_ball_sqrt_est1],
[f3_swpx_angle_ball_sqrt_est1, f3_swpy_angle_ball_sqrt_est1, f3_swpz_angle_ball_sqrt_est1]])

print("Matrix for eigenvalues ",F3_switchball_angle_sqrt_est1)

JacobianF_falling_angle_ball_sqrt_est1 = Matrix(F3_switchball_angle_sqrt_est1.subs([(x,5),(y,5),(z,0)]))

#eigenvalues
jaceiglist_sqrt_est1 = list(JacobianF_falling_angle_ball_sqrt_est1.eigenvals())

print(F3_switchball_angle_sqrt_est1,"Jacobian Matrix")
print(F3_switchball_angle_sqrt_est1.subs([(x,5),(y,5),(z,0)],"Substited values"))
print(F3_switchball_angle_sqrt_est1.eigenvals(),"Eigenvalues")

#stability step function pattern
def hstabilitygetting_angle_ball_sqrt_est1(MatrixForYacobian, values):

    JacobianF_falling_angle_ball_step1 = Matrix(
    MatrixForYacobian.subs([(x, values[0]), (y, values[1]), (z, values[2])]))
    jaceiglist = list(JacobianF_falling_angle_ball_step1.eigenvals())

    eigenlist = []

    for i in range(len(jaceiglist)):
        eq = sympy.S(str(jaceiglist[i]))
        eq = eq.subs(sympy.Function('arctg'), sympy.atan)
        eq = eq.subs(sympy.Function('sqrt'), math.sqrt)
        eq = (((eq).evalf()))
        eigenlist.append(eq)

    return 2 / (reduce(lambda x, y: abs(x) if abs(x) > abs(y) else abs(y), eigenlist))

jaceiglist_falling_angle_ball_sqrt_est1 = list(JacobianF_falling_angle_ball_sqrt_est1.eigenvals())
hbegin_falling_ball_angle_ball_sqrt_est1 = hstabilitygetting_angle_ball_sqrt_est1(F3_switchball_angle_sqrt_est1, [5,5,0])/40

print("Starting stability step for angle ball extended ODE", hbegin_falling_ball_angle_ball_sqrt_est1)
time.sleep(5)

establish_sqrt_est1_esposito_on = False
#'''##Need integration extended ODE falling ball
iterations_establ_angle_sqrt_est1,endtime_establ_angle_sqrt_est1, yprint_s_establ_angle_sqrt_est1 = rkf2stepcontrol(matrix_size_sqrt_est1,
                                                                                                                    dydx_Angleball_sqrt_est1
                                                                                                                    , A_Angleball_sqrt_est1,
                                                                                                                    ystart_Angleball_sqrt_est1,
                                                                                                                    b2x2, dxdt_y_establ_sqrt_est1,
                                                                                                                    hbegin_falling_ball_angle_ball_sqrt_est1,
                                                                                                                    0
                                                                                                                    , 2,
                                                                                                                    0.000001,
                                                                                                                    predicate_function_establ,
                                                                                                                    F3_switchball_angle_sqrt_est1,
                                                                                                                    hstabilitygetting_angle_ball_sqrt_est1,
                                                                                                                    0.005,establish_sqrt_est1_esposito_on)

#xprint_s_establ_angle_step1(0)
print("Shape of integrated matrix",yprint_s_establ_angle_sqrt_est1.shape)
yprint_s_establ_angle_sqrt_est1 = numpy.delete(yprint_s_establ_angle_sqrt_est1, 0, -1)
print("Shape after deleting first column",yprint_s_establ_angle_sqrt_est1.shape)

endtime_angle_ball_for_step2 = yprint_s_establ_angle_sqrt_est1[2][-1]
print("Integration end time",endtime_angle_ball_for_step2)

#argument1 = list(reversed(yprint_s_establ_angle_step1[0]))
#argument2 = list(reversed(yprint_s_establ_angle_step1[1]))
#argument3 = list(reversed(yprint_s_establ_angle_step1[2]))

argument1 = list((yprint_s_establ_angle_sqrt_est1[0]))
argument2 = list((yprint_s_establ_angle_sqrt_est1[1]))
argument3 = list((yprint_s_establ_angle_sqrt_est1[2]))

print('Amount of iterations',iterations_establ_angle_sqrt_est1)
print('Время окончания', endtime_establ_angle_sqrt_est1)

t = linspace(0, float(endtime_establ_angle_sqrt_est1), yprint_s_establ_angle_sqrt_est1.shape[1])

title('Establishing method falling ball step 1 Vy"=--25*sqrt(y)')
plot(t, argument1, '-o', linewidth=2)
plot(t, argument2, '-o', linewidth=2)
plot(t, argument3, '-o', linewidth=2)
legend(["y", "Vy", "t"], loc ="upper right")
print()
ylabel("argument")
xlabel("t")
grid(True)
show() # display
#'''#Establishing Method sqrt step1 end


#'''#Establishing Method sqrt step2 begin
#Establishing Method for original
#endtime_angle_ball_for_step2 = 2.37349620145370
matrix_size_sqrt_est2 = 2
k_angle_ball = 3
g = 9.81
A_Angleball_sqrt_est2 = Matrix([k_angle_ball, g])
ystart_Angleball_sqrt_est2 = Matrix([[5.], [5.]])
dydx_Angleball_sqrt_est2 = zeros(matrix_size, 1)

def dxdt_y_sqrt_est2(A_Angleball_sqrt_est2, ystart_Angleball_sqrt_est2, b2x2):


    dydx_Angleball_sqrt_est2[0, 0] = ystart_Angleball_sqrt_est2[1, 0]
    dydx_Angleball_sqrt_est2[1, 0] = (-25 * math.sqrt(ystart_Angleball_sqrt_est2[0, 0]))

    return dydx_Angleball_sqrt_est2



J1_switchball_angle_ball_step2 = sympy.Function('J')(x, y)

#setting Jacobian Matrix
f1_swpx_angle_ball_sqrt_est2 = '0'
f1_swpy_angle_ball_sqrt_est2 = '1'
f2_swpx_angle_ball_sqrt_est2 = '0'
f2_swpy_angle_ball_sqrt_est2 = '-25/sqrt(y)'

F3_switchball_angle_sqrt_est2 = Matrix([[f1_swpx_angle_ball_sqrt_est2, f1_swpy_angle_ball_sqrt_est2],
                                    [f2_swpx_angle_ball_sqrt_est2, f2_swpy_angle_ball_sqrt_est2]])

JacobianF_falling_angle_ball_sqrt_est2 = Matrix(F3_switchball_angle_sqrt_est2.subs([(x,5),(y,5)]))
print('Jacobian Matrix for original ODE', JacobianF_falling_angle_ball_sqrt_est2)
jaceiglist = list(JacobianF_falling_angle_ball_sqrt_est2.eigenvals())
print('Eigenvalues original ODE', jaceiglist)

#stability function for original ODE
def hstabilitygetting_angle_ball_sqrt_est2(MatrixForYacobian, values):
    JacobianF_falling_angle_ball_sqrt_est2 = Matrix(
    MatrixForYacobian.subs([(x, values[0]), (y, values[1])]))
    jaceiglist_sqrt_est2 = list(JacobianF_falling_angle_ball_sqrt_est2.eigenvals())

    eigenlist = []

    for i in range(len(jaceiglist_sqrt_est2)):
        eq = sympy.S(str(jaceiglist[i]))
        eq = eq.subs(sympy.Function('arctg'), sympy.atan)
        eq = eq.subs(sympy.Function('sqrt'), math.sqrt)
        eq = float((eq).evalf())
        eigenlist.append(eq)

    return  2 / (reduce(lambda x, y: abs(x) if abs(x) > abs(y) else abs(y), eigenlist))



jaceiglist_falling_angle_ball_sqrt_est2 = list(JacobianF_falling_angle_ball_sqrt_est2.eigenvals())


hbegin_falling_ball_angle_ball_sqrt_est2 = hstabilitygetting_angle_ball_sqrt_est2(F3_switchball_angle_sqrt_est2, [5.,5.])/4000
print("Beginning step or original ODE by stability", hbegin_falling_ball_angle_ball_sqrt_est2)



establish_sqrt_est2_esposito_on = False
#'''##Need integration original ODE angle ball
iterations_establ_angle_sqrt_est2,endtime_establ_angle_sqrt_est2, yprint_s_establ_angle_sqrt_est2 = rkf2stepcontrol(matrix_size_sqrt_est2,
                                                                                    dydx_Angleball_sqrt_est2
                                                                                    , A_Angleball_sqrt_est2,
                                                                                    ystart_Angleball_sqrt_est2,
                                                                                    b2x2, dxdt_y_sqrt_est2,
                                                                                    hbegin_falling_ball_angle_ball_sqrt_est2,
                                                                                    0
                                                                                    , endtime_angle_ball_for_step2,
                                                                                    #time from extended ODE
                                                                                    0.00001,
                                                                                    predicate_function_establ,
                                                                                    F3_switchball_angle_sqrt_est2,
                                                                                    hstabilitygetting_angle_ball_sqrt_est2,
                                                                                    0.001,establish_sqrt_est2_esposito_on)




yprint_s_establ_angle_sqrt_est2 = numpy.delete(yprint_s_establ_angle_sqrt_est2, 0, -1)
argument1 = list((yprint_s_establ_angle_sqrt_est2[0]))
argument2 = list((yprint_s_establ_angle_sqrt_est2[1]))

print('Amount of iterations',iterations_establ_angle_sqrt_est2)
print('Time of end', endtime_establ_angle_sqrt_est2)

t = linspace(0, float(endtime_establ_angle_sqrt_est2), yprint_s_establ_angle_sqrt_est2.shape[1])


title('Establishing method falling ball step 2 Vy"=-25*sqrt(y)')
plot(t, argument1, '-o', linewidth=2)
plot(t, argument2, '-o', linewidth=2)
legend(["y", "Vy", "t"], loc ="upper right")
print()
ylabel("argument")
xlabel("t")
grid(True)
show() # display
#'''#Establishing Method sqrt step2 end

#Bisection sqrt part start

hn_sqrt = 0.001

Aklinnonlin_sqrt = Matrix([[k, 1], [0, -0.5]])

dydxbisect_sqrt = zeros(matrix_size, 1)

print(dydxbisect_sqrt)

xstartswitchpoint_sqrt = Matrix([[5.], [5.]])

def dxdt_y_sqrt(Aklinnonlin_sqrt, xstartswitchpoint_sqrt, b2x2):

    dydxbisect[0, 0] = xstartswitchpoint_sqrt[1, 0]
    dydxbisect[1, 0] = -25*math.sqrt (xstartswitchpoint_sqrt[0, 0])  ##Vy' = -k*Vy-g
    return dydxbisect

k1_sqrt = zeros(matrix_size, 1)
k2_sqrt = zeros(matrix_size, 1)
k3_sqrt = zeros(matrix_size, 1)
k2k1norm_sqrt = zeros(matrix_size, 1)

xprint_s_bisect_sqrt= Matrix([5.,5.])
xstartswitchpoint_sqrt = Matrix([5.,5.])
iterations2_sqrt = 0
matrix_size_sqrt = 2
t2 = 0

def predicate_function_linear_xsecond(xstart,eps):
    return float(xstart[0]) < 0 + eps

#'''#Bisection sqrt
start_time_sqrt = datetime.now()
while True:  # Integrate
    for i in (range(matrix_size_sqrt)):
        iterations2_sqrt += 1
        k1_sqrt[i, 0] = hn_sqrt * dxdt_y_sqrt(Aklinnonlin_sqrt, xstartswitchpoint_sqrt, b2x2)[i, 0]
        k2_sqrt[i, 0] = hn_sqrt * dxdt_y_sqrt(
            Aklinnonlin_sqrt,
            xstartswitchpoint_sqrt + k1_sqrt[i, 0] * sympy.ones(*xstartswitchpoint_sqrt.shape),
            b2x2,
        )[i, 0]
        xstartswitchpoint_sqrt[i, 0] = xstartswitchpoint_sqrt[i, 0] + 0.5 * (k1_sqrt[i, 0] + k2_sqrt[i, 0])
        xprint_s_bisect_sqrt = xprint_s_bisect_sqrt.col_insert(1, Matrix([xstartswitchpoint_sqrt]))
    dxdt_y(Aklinnonlin_sqrt, xstartswitchpoint_sqrt, b2x2)
    print("Integrated parameter, Runge-Kutta Bisection",xstartswitchpoint_sqrt)
    print(t2)

    t2 += hn_sqrt
    if predicate_function_linear_xsecond(xstartswitchpoint_sqrt,epsilon_eiler_sqrt/1000):
        break
    end_time = datetime.now()
    print('Duration Bisection: {}'.format(end_time - start_time))

xprint_s_bisect_sqrt.col_del(0)  # Delete first column

argument1 = list(reversed(xprint_s_bisect_sqrt.row(0)))
argument2 = list(reversed(xprint_s_bisect_sqrt.row(1)))

t = linspace(0, t2, iterations2_sqrt)

title('Bisection falling ball sqrt')
plot(t, argument1, '-o', linewidth=2)
plot(t, argument2, '-o', linewidth=2)
legend(["y", "Vy"], loc ="upper right")
print()
ylabel("argument")
xlabel("t")
grid(True)
show() # display

#'''#Bisection sqrt part end
