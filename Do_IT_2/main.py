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

print('Шаг по устойчивости',hstability)



def rkf2stepcontrol(
        matrix_size,
        # k1,k2,k3,k2k1norm,
        dydx, A, x, b, dxdt, hn, t, tout, eps, predicate_func, MatrixForYacobian, hstabilitygetting, halgebraic):
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

        while ((predicate_func(x, eps) == False) or (t - tout > eps)):
            b = t
            iterations += 1
            print("Я вошёл в ", x)

            xprint_s = numpy.c_[xprint_s, x]
            k1 = hn * dxdt(A, x, b)

            for i in (range(matrix_size)):
                k2[i, 0] = hn * dxdt(A, x + k1[i, 0] * numpy.ones(x.shape,dtype=float,order='C'), b)[i, 0]

            xold = x

            for i in (range(matrix_size)):
                x[i, 0] = x[i, 0] + 0.5 * (k1[i, 0] + k2[i, 0])

            hstability = 0
            hacc = 0
            hbuffer = 0

            norm = float(0.5 * (k2 - k1).norm())
            if (0.5 * (k2 - k1).norm()) < eps:
                hstability = hstabilitygetting(MatrixForYacobian, x)
                #print("New stability step", hstability)
                # MatrixForYacobian

            elif (norm >= eps):  # Recalculate accuracy step
            #if (True):
                hneeded = sympy.symbols('hn')

                j = 0

                for j in range(matrix_size):  # автоматический подсчёт разности компонент

                    # k2k1norm[j, 0] = float(dxdt(dydx, A, x + k1[i, 0] * sympy.ones(*x.shape), b)[j, 0]) - float(dxdt(dydx, A, x, b)[j, 0])
                    k2k1norm[j, 0] = float(dxdt(A, x + k1[i, 0] * numpy.ones(x.shape,dtype=float,order='C'), b)[j, 0]) - \
                                     float(dxdt(A, x, b)[j, 0])

                epspart = str(2 * eps)
                #print("Норма матрицы",k2k1norm)

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

                hacclist = list(hacc)
                #for i in range(len(hacclist)):
                    #print("Значение", hacclist[i])

                if hacc.is_empty == True:
                    hbuffer = hn
                elif (hacc.sup) <= 0:
                    hbuffer = 0.000001
                #    print("Accuracy step is zero or negative")
                elif ((hacc.sup) <= 0.0000000001):
                    hbuffer = 0.000001
                else:
                    hbuffer = float(max(hacc))

                #print(float(max(hacc)), 'Шаг по точности')

                hn = hbuffer
                # print(float(hacc))

            hn = min(halgebraic, hn)  # Minimal step from old step and step by event function
            #print('New step =', hn)

            hnp = min(hn, min(hbuffer, hstability))  # makes minimal step from accuracy, stability and old step
            print("Current step",hn)
            # hn = hnp

            t += hn
            print("Current time", t)
            # Next stage calculating
            for i in range(matrix_size):

                k3[i, 0] = hn * dxdt(A, x, b)[i, 0]

            endtime = t;

    end_time = datetime.now()
    print('Duration: {}'.format(end_time - start_time))
    return iterations, endtime, xprint_s


matrix_size = 3
k_angle_ball = 3
g = 9.81

A_Angleballbeforeeps = Matrix([k_angle_ball, g])  # смена Numpy матрицы на Sympy
ystart_Angleballbeforeeps = Matrix([[5.], [5.],[0.]])  # xx,VV            #смена Numpy матрицы на Sympy
dydx_Angleballbeforeeps = zeros(matrix_size, 1)

def dxdt_y_establstep1(A_Angleballbeforeeps, ystart_Angleballbeforeeps, b2x2):  # система уравнений для проверки поиска точки переключения

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

    #print("Собственные числа", eigenlist)
    return  2 / (reduce(lambda x, y: abs(x) if abs(x) > abs(y) else abs(y), eigenlist))



jaceiglist_falling_angle_ball_step1 = list(JacobianF_falling_angle_ball_step1.eigenvals())
hbegin_falling_ball_angle_ball_step1 = hstabilitygetting_angle_ball_step1(F3_switchball_angle_step1, [5,5,0])/400


print("Starting stability step for angle ball extended ODE", hbegin_falling_ball_angle_ball_step1)
time.sleep(5)


#'''##Need integration extended ODE falling ball
iterations_establ_angle_step1,endtime_establ_angle_step1, yprint_s_establ_angle_step1 = rkf2stepcontrol(matrix_size,
                                                                                    dydx_Angleballbeforeeps
                                                                                    , A_Angleballbeforeeps,
                                                                                    ystart_Angleballbeforeeps,
                                                                                    b2x2, dxdt_y_establstep1,
                                                                                    hbegin_falling_ball_angle_ball_step1,
                                                                                    # НА СТАЦИОНАР,КАК НА КАРТИНКЕ
                                                                                    # hbegin_falling_ball_step1_alt,
                                                                                    0
            #                                , 20, 0.1,predicate_function_establ_step1, F3, hstabilitygetting) #Старое
                                                                                    , 3.5,
                                                                                    # ВРЕМЯ, ЗА КОТОРОЕ ВЫЙДЕМ НА СТАЦИОНАР
                                                                                    0.00001,  # ТРЕБУЕМАЯ ТОЧНОСТЬ
                                                                                    # 0.000001, ДОЛГО
                                                                                    # 0.01, ЗАКАНЧИВАЕТ В 3.63, НЕ В НУЛЕ
                                                                                    # 0.9,
                                                                                    predicate_function_establ,
                                                                                    F3_switchball_angle_step1,
                                                                                    hstabilitygetting_angle_ball_step1,0.001)





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
A_Angleballaftereps = Matrix([k_angle_ball, g])  # смена Numpy матрицы на Sympy
ystart_Angleballaftereps = Matrix([[5.], [5.]])  # xx,VV            #смена Numpy матрицы на Sympy
dydx_Angleballaftereps = zeros(matrix_size, 1)

def dxdt_y_establstep2(A_Angleballaftereps, ystart_Angleballaftereps, b2x2):  # система уравнений для проверки поиска точки переключения


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
                                                                                    0.0000001,  # ТРЕБУЕМАЯ ТОЧНОСТЬ
                                                                                    predicate_function_establ,
                                                                                    F3_switchball_angle_step2,
                                                                                    hstabilitygetting_angle_ball_step2,0.001)





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
        print("Inside Eiler angle ball")
        print("Current time", endtime_eilermethodvariablestep)
        ystartlinearelersecondvarstep[2, 0] = ystartlinearelersecondvarstep[0, 0]  # g = x

        hpeilersecondvarstep_angle_ball = (gammaeilersecond_angle_ball - 1) * ystartlinearelersecondvarstep[2, 0] \
                               / (-3 * VYold - 9.81 + VYold)  # hp = (gamma-1)*g/(-omega**2*xold+Vxold)
        #There must be event control function  (gamma-1)*diffentials of needed original ODE



        event_vector = ystartlinearelersecondvarstep[2, 0] / (-3 * VYold - 9.81 + VYold)
        print("Hp by event func  and h input by user",hpeilersecondvarstep_angle_ball, heilersecond_angle_ball)
        hweilersecondvarstep_angle_ball = min(hpeilersecondvarstep_angle_ball, heilersecond_angle_ball)  # hw = hp
        print("Step after", hweilersecondvarstep_angle_ball)
        endtime_eilermethodvariablestep = endtime_eilermethodvariablestep + hweilersecondvarstep_angle_ball

        Fi = Matrix([VYold, -3  * VYold - 9.81])
        ystartlinearelersecondvarstep[0, 0] = Yold + hweilersecondvarstep_angle_ball * Fi[0, 0]  # x = xold+hw*Fi[1]
        ystartlinearelersecondvarstep[1, 0] = VYold + hweilersecondvarstep_angle_ball * Fi[1, 0]  # vx = vxold+hw*Fi[2]
        Yold = ystartlinearelersecondvarstep[0, 0]  # xold = x
        VYold = ystartlinearelersecondvarstep[1, 0]  # vxold = vx
        ystartlineareler_discrete_print_variablstep = \
            ystartlineareler_discrete_print_variablstep.col_insert(1, Matrix([ystartlinearelersecondvarstep]))
        print("Integrated parameter", ystartlinearelersecondvarstep)

        h_for_output.append(hweilersecondvarstep_angle_ball)
        event_vect.append(event_vector)
        hweeee = hweeee.col_insert(1, Matrix([hweilersecondvarstep_angle_ball]))
        print("Step vector",hweeee)
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

print('Время окончания', endtime_eiler_angle_ball)

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
print("Шаг на основе метода Эйлера", hweilerbasedrungekutta)

#'''#МЕТОД ЭЙЛЕРА

#hweilerbasedrungekutta = 1.39E-5

def predicate_function_linear_xsecond(xstart,eps):

    return float(xstart[0]) < 0 + eps


#'''# Esposito method step control by event function, step got from Eiler method
iterations_rkeiler_angle_step2,endtime_rkeiler_angle_step2, yprint_s_rkeiler_angle_step2 = rkf2stepcontrol(matrix_size,
                                                                                    dydx_Angleballaftereps
                                                                                    , A_Angleballaftereps,
                                                                                    ystart_Angleballaftereps,
                                                                                    b2x2, dxdt_y_establstep2,
                                                                                       hweilerbasedrungekutta,
                                                                                    0
                                                                                    , 10,
                                                                                    epsilon_eiler_second_angle_ball,
                                                                                    #needed accuracy
                                                                                    predicate_function_linear_xsecond,
                                                                                    F3_switchball_angle_step2,
                                                                                    hstabilitygetting_angle_ball_step2,
                                                                                    hweilerbasedrungekutta)


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