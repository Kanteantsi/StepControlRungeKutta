# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
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

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/


print(max(-0.5*math.sqrt(1.0*math.atan(5)**2 - 0.399230769230769*math.atan(5) + 0.910556360946746) + 0.5*math.atan(5) - 0.477115384615385, 0.5*math.sqrt(1.0*math.atan(5)**2 - 0.399230769230769*math.atan(5) + 0.910556360946746) + 0.5*math.atan(5) - 0.477115384615385, 0))

x, y, z = sympy.symbols("x y z")

b2x2 = Matrix([[5.], [-3.]])
#jaceiglist = sympy.simplify("-0.5*sqrt(1.0*arctg(5)**2 - 0.399230769230769*arctg(5) + 0.910556360946746) + 0.5*arctg(5) - 0.477115384615385, 0.5*sqrt(1.0*arctg(5)**2 - 0.399230769230769*arctg(5) + 0.910556360946746) + 0.5*arctg(5) - 0.477115384615385, 0")
#print(jaceiglist)

#print(sympy.symplify("arctg(5)").evalf)

#print(sympy.sympify("sympy.arctg(5)"))


#print(float(sympy.sympify("arctg(5)").evalf()))



k = 3
g = 9.81
#X = sympy.Matrix(["y*math.arctg(x)", "(-k*y-g)*arctg(x)"," math.arctg(x)"])
#Y =  sympy.Matrix([x, y,z])

#print (X.jacobian(Y))

#jaceiglist = tuple(JacobianF_falling_angle_ball_step1.eigenvals())
#...  .subs(Function('arctg'), atan)
x, y, z = sympy.symbols("x y z")
#hweilerbasedrungekutta = 1.39E-5

def predicate_function_linear_xsecond(xstart,eps):
    #return xstart[0, 0] < 0
    #return xstart[0, 0] < 0+0.0001
    #return xstart[0,0] < 0 + 0.0001
    return float(xstart[0]) < 0 + eps


def predicate_function_establ(xstart,eps):
    #return xstart[0, 0] < 0
    #return xstart[0, 0] < 0+0.0001
    #return xstart[0,0] < 0 + 0.0001
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

'''
eq = sympy.S(str(jaceiglist[0]))
eq = eq.subs(sympy.Function('arctg'), sympy.atan)
eq = eq.subs(sympy.Function('sqrt'), math.sqrt)
eigenlist.append(eq)
print("Значения собственных чисел", eigenlist)
'''

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
    with open(__file__ + 'Вывод RK2.txt', 'w') as f:
        global needpredfunc
        start_time = datetime.now()

        #xbefprint = numpy.zeros(matrix_size, 1)
        xprint_s = numpy.zeros((matrix_size, 1))
        #newxfork2 = numpy.zeros(matrix_size, 1)

        k1 = numpy.zeros((matrix_size, 1))
        k2 = numpy.zeros((matrix_size, 1))
        k3 = numpy.zeros((matrix_size, 1))
        k2k1norm = numpy.zeros((matrix_size, 1))
        #print("Тип икса",x.type)
        #while ((predicate_func(x,eps) == False) or (t - tout > eps)):
        print("Условие 1",(predicate_func(x, eps) == False))
        print("Условие 2", tout - t > eps)
        while ((predicate_func(x, eps) == False) or (t - tout > eps)):
            print("Условие 1", (predicate_func(x, eps) == False))
            print("Условие 2", tout - t > eps)
            #print("Предикат",predicate_func(x,eps))
            #print('Текущее время', t)
            b = t
            iterations += 1
            print("Я вошёл в ", x)

            xprint_s = numpy.c_[xprint_s, x]
            #xprint_s = xprint_s.col_insert(1, Matrix([x]))
            #xprint_s = xprint_s
            k1 = hn * dxdt(A, x, b)

            #print(k1, 'k1')


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
                #print("Новый шаг по устойчивости", hstability)
                # MatrixForYacobian


            # Чисто теоретически мы должны перебирать матрицу Якоби на каждом шагу,
            # подставляя актуальные вычисленные значения в неё, тем самым вычисляя максимальное
            # по модулю собственное число на каждом шаге, и получая шаг


            elif (norm >= eps):  # пересчёт шага по точности
            #if (True):  # пересчёт шага по точности
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
                        k2k1norm[i,0]) + ')**2 +'  # автоматическая подстановка шаг*разность компоненты в квадрате
                    i += 1
                for i in range(matrix_size - 1, matrix_size):
                    equationstring = equationstring + '(hn*' + str(
                        k2k1norm[i,0]) + ')**2'  # автоматическая подстановка шаг*разность компоненты в квадрате
                    i += 1
                equationstring = equationstring + ') - ' + epspart
                #print(equationstring, 'Уравнение')
                hacc = sympy.solveset(equationstring, hneeded)
                #hacc = sympy.solveset(equationstring, hneeded, domain=S.Reals)

                #print('решение', hacc)
                #print(type(hacc))

                hacclist = list(hacc)
                #for i in range(len(hacclist)):
                    #print("Значение", hacclist[i])

                if hacc.is_empty == True:
                    hbuffer = hn
                elif (hacc.sup) <= 0:
                    hbuffer = 0.000001
                #    print("Шаг по точности равен 0 или отрицателен")
                elif ((hacc.sup) <= 0.0000000001):
                    hbuffer = 0.000001
                else:
                    hbuffer = float(max(hacc))

                #print(float(max(hacc)), 'Шаг по точности')

                hn = hbuffer
                # print(float(hacc))

            hn = min(halgebraic, hn)  # МИНИМАЛЬНЫЙ ШАГ ИЗ ТОЧНОСТИ, УСТОЙЧИВОСТИ И АЛГЕБРАИЧЕСКОГО ПО МЕТОДУ
            #print('Новый шаг =', hn)

            # print('Старый шаг',hn,'Шаг по устойчивости',hstability,'Шаг по точности',hacc)
            #print('Шаги', hn, hbuffer, hstability)
            hnp = min(hn, min(hbuffer, hstability))  # выдаёт говно, при max не уходит начальный огромный шаг
            print("Текущий шаг",hn)
            # при min обращает шаг в 0, не давая продвижения
            # print('Шаг по устойчивости и точности',hnp)
            # hn = hnp


            #t += hn*100
            t += hn
            print("Текущее время", t)
            # обновлённый код расчёта третьей стадии, нужен для контроля шага по устойчивости
            for i in range(matrix_size):

                k3[i, 0] = hn * dxdt(A, x, b)[i, 0]

            # print('Время моделирования: ',t)
            #global endtime
            endtime = t;
            #break

        #break
    # print('Форма выходного вектора', xprint_s.shape)
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

    # print('Shape of argument ',xstartswitchpoint.shape)
    # print('Shape of derivative ', dydx.shape)
    # print(type(xstartswitchpoint), 'Аргумент')
    # print(type(dydx[0, 0]), 'Производная')
    #dydx_Angleballbeforeeps[0, 0] = ystart_Angleballbeforeeps[1, 0]*ystart_Angleballbeforeeps[0, 0]*9.81
    dydx_Angleballbeforeeps[0, 0] = ystart_Angleballbeforeeps[1, 0] * math.atan(ystart_Angleballbeforeeps[0, 0])
    dydx_Angleballbeforeeps[1, 0] = (-1 * A_Angleballbeforeeps[0, 0] * (ystart_Angleballbeforeeps[1, 0]) - 9.81)* math.atan(ystart_Angleballbeforeeps[0, 0])
    dydx_Angleballbeforeeps[2, 0] = math.atan(ystart_Angleballbeforeeps[0, 0])
    # dydx[0, 0] = sum([dydx[0, 0], xstartswitchpoint[1, 0]])  # put args in list
    #dydx_Angleballbeforeeps[1, 0] = (-1 * A_Angleballbeforeeps[0, 0] * (ystart_Angleballbeforeeps[1, 0]) - 9.81)*(ystart_Angleballbeforeeps[0, 0]*9.81)  ##Vy' = -k*Vy-g
    #dydx_Angleballbeforeeps[2,0] = ystart_Angleballbeforeeps[0, 0]*9.81

    # print(dydx)
    return dydx_Angleballbeforeeps


J1_switchball_angle_ball_step1 = sympy.Function('J')(x, y, z)


f1_swpx_angle_ball_step1 = 'arctg(x)'  # x’ =  V*y**5*g          x' = x^5*y*g
print(f1_swpx_angle_ball_step1)
f1_swpy_angle_ball_step1 = 'y/(x*x+1)'  # x’ =  V*y**5*g          x' = x^5*y*gХ
print(f1_swpx_angle_ball_step1)
f1_swpz_angle_ball_step1 = '0'  # x’ =  V*y**5*g          x' = x^5*y*g
print(f1_swpx_angle_ball_step1)
f2_swpx_angle_ball_step1 = '-3*arctg(x)' # y'= (-k*y-9.81)*x^5*y*g
print(f2_swpx_angle_ball_step1)
f2_swpy_angle_ball_step1 = '-3*(x+3.27)/(y*y+1)'  # y'= (-k*y-9.81)*x^5*y*g
print(f2_swpy_angle_ball_step1)
f2_swpz_angle_ball_step1 = '0'       # y'= (-k*y-9.81)*x^5*y*g
print(f2_swpy_angle_ball_step1)
f3_swpx_angle_ball_step1 = "1/(x*x+1)"        #z' = x^5*y*g
print(f2_swpy_angle_ball_step1)
f3_swpy_angle_ball_step1 = "0"        #z' = x^5*y*g
print(f2_swpy_angle_ball_step1)
f3_swpz_angle_ball_step1 = "(0)"        #z' = x^5*y*g
print(f3_swpz_angle_ball_step1)

'''
f1_swpx_angle_ball_step1 = 'arctg(x)'  # x’ =  V*y**5*g          x' = x^5*y*g
print(f1_swpx_angle_ball_step1)
f1_swpy_angle_ball_step1 = 'y/(x*x+1)'  # x’ =  V*y**5*g          x' = x^5*y*g
print(f1_swpx_angle_ball_step1)
f1_swpz_angle_ball_step1 = '0'  # x’ =  V*y**5*g          x' = x^5*y*g
print(f1_swpx_angle_ball_step1)
f2_swpx_angle_ball_step1 = '-3*arctg(x)' # y'= (-k*y-9.81)*x^5*y*g
print(f2_swpx_angle_ball_step1)
f2_swpy_angle_ball_step1 = '-3*(x+3.27)/(y*y+1)'  # y'= (-k*y-9.81)*x^5*y*g
print(f2_swpy_angle_ball_step1)
f2_swpz_angle_ball_step1 = '0'       # y'= (-k*y-9.81)*x^5*y*g
print(f2_swpy_angle_ball_step1)
f3_swpx_angle_ball_step1 = "1/(x*x+1)"        #z' = x^5*y*g
print(f2_swpy_angle_ball_step1)
f3_swpy_angle_ball_step1 = "0"        #z' = x^5*y*g
print(f2_swpy_angle_ball_step1)
f3_swpz_angle_ball_step1 = "(0)"        #z' = x^5*y*g
print(f3_swpz_angle_ball_step1)
'''

#F3_switchball_angle_step1 = Matrix([f1_swpx_angle_ball_step1, f2_swpy_angle_ball_step1, f3_swpz_angle_ball_step1])
#'''
F3_switchball_angle_step1 = Matrix([[f1_swpx_angle_ball_step1, f1_swpy_angle_ball_step1, f1_swpz_angle_ball_step1],
                                    [f2_swpx_angle_ball_step1, f2_swpy_angle_ball_step1, f2_swpz_angle_ball_step1],
                                    [f3_swpx_angle_ball_step1, f3_swpy_angle_ball_step1, f3_swpz_angle_ball_step1]])
#'''
# print("Якобиан-1 переключения ", F2.jacobian([x, y]))

'''
allo = \
    Matrix([
        #[5*x**4, 1, 0],
        [math.atan(y), x/(y*y+1), 0],
        [-3*math.atan(y),-3*(x+3.27)/(y*y+1), 0],
        [1/(x*x+1),0,0]
        #[-5*g * x**4*(A_Angleballbeforeeps[0, 0]*y+g), g*x**5*(-2*A_Angleballbeforeeps[0, 0]*y-g), 0],
        #[5*x**4, 1, 0]
    ])
# print("Якобиан-2 переключения ", F2.jacobian([x, y]).subs([(x, 5), (y, 5)]))
print("Матрица для собственных чисел ",allo)
'''

print("Матрица для собственных чисел ",F3_switchball_angle_step1)
print("Форма матрицы для собственных чисел мяч под углом",F3_switchball_angle_step1.shape)


#JacobianF_falling_angle_ball_step1 = Matrix(F3_switchball_angle_step1.jacobian([x, y, z]).subs([(x, 5.), (y, 4.9), (z, 0.)]))
JacobianF_falling_angle_ball_step1 = Matrix(F3_switchball_angle_step1.subs([(x,5),(y,5),(z,0)]))

print("Форма матрицы для собственных чисел мяч под углом",F3_switchball_angle_step1.shape)

print('Якоби 3x3 Мяч под углом ', JacobianF_falling_angle_ball_step1)
jaceiglist = list(JacobianF_falling_angle_ball_step1.eigenvals())
print('Собственные числа,3x3 Мяч под углом', jaceiglist)


#for i in range(len(jaceiglist)):
#jaceiglist = sorted(jaceiglist, key = math.abs(jaceiglist[i]))


#jaceiglist = (sorted(jaceiglist, key=abs, reverse=True))

#print('Собственные числа,3x3 Мяч под углом', jaceiglist)


#print('Шаг равен Мяч под углом', 2 / (reduce(lambda x, y: abs(x) if abs(x) > abs(y) else abs(y), jaceiglist)))


print(F3_switchball_angle_step1,"Сама матрица для якоби")
print(F3_switchball_angle_step1.subs([(x,5),(y,5),(z,0)],"Подставленные значения"))
print(F3_switchball_angle_step1.eigenvals(),"Собств значения")
#print(F3_switchball_angle_step1.subs([(x, values[0]), (y, values[1]), (z, values[2])]))
#print(F3_switchball_angle_step1.jacobian([x, y, z]),"Якоби мяч под углом")


def hstabilitygetting_angle_ball_step1(MatrixForYacobian, values):
    #MatrixForYacobian = Matrix([MatrixForYacobian])
    #print("Форма матрицы для Якоби",MatrixForYacobian.shape)
    JacobianF_falling_angle_ball_step1 = Matrix(
    MatrixForYacobian.subs([(x, values[0]), (y, values[1]), (z, values[2])]))
        #MatrixForYacobian.jacobian.subs([(x, values[0]), (y, values[1]), (z, values[2])))
    #print("Форма матрицы для Якоби после подстановки", JacobianF_falling_angle_ball_step1.shape)
    # print('Якоби 3x3', JacobianF_falling_ball_step1)
    #jaceiglist = Tuple(*JacobianF_falling_angle_ball_step1.eigenvals())
    #eq = S(jaceiglist)
    #eq.subs((Function('arctg'), sympy.atan))
    #print("Тест",eq)
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


# def hstabilitygetting_falling_ball_step1(MatrixForYacobian, values):
# JacobianF = Matrix(MatrixForYacobian.jacobian([x, y, z]).subs([(x, values[0]), (y, values[1]), (z, values[2])]))
# jaceiglist = list(JacobianF.eigenvals())
# return 2 / (reduce(lambda x, y, z: d=abs(a) if abs(a), jaceiglist))

jaceiglist_falling_angle_ball_step1 = list(JacobianF_falling_angle_ball_step1.eigenvals())
# hbegin_falling_ball_step1 = 2 / (reduce(lambda x, y, z : abs(x) if (abs(x) > abs(y) and abs(x) > abs(z))
#                        else abs(y) if (abs(y) > abs(z)) else z , jaceiglist_falling_ball_step1))


hbegin_falling_ball_angle_ball_step1 = hstabilitygetting_angle_ball_step1(F3_switchball_angle_step1, [5,5,0])/400


print("Начальный шаг по устойчивости для первого шага мяча 3x3 Мяч под углом", hbegin_falling_ball_angle_ball_step1)
time.sleep(5)



'''
print("Предикат в 2.36429961675833",predicate_function_linear_xsecond([[0.0231390146588621], [-3.26312775428630]]))
print("Предикат в 2.37065257029006",predicate_function_linear_xsecond([[0.00240795905965600], [-3.26325749121318]]))
print("Предикат в 2.37138934608258",predicate_function_linear_xsecond([[3.64784119559927e-6], [-3.26327237790665]]))
print("Предикат в 2.37140324751262",predicate_function_linear_xsecond([[-4.17166268049767e-5], [-3.26327265847151]]))
'''


#'''##Требует интегрирования МЕТОД УСТАНОВЛЕНИЯ МЯЧ ПОД УГЛОМ ШАГ 1
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
print("Форма наинтегрированного",yprint_s_establ_angle_step1.shape)
yprint_s_establ_angle_step1 = numpy.delete(yprint_s_establ_angle_step1, 0, -1)
print("Форма наинтегрированного",yprint_s_establ_angle_step1.shape)

#yprint_s_establ_angle_step1.col_del(0)

endtime_angle_ball_for_step2 = yprint_s_establ_angle_step1[2][-1]
print("Время окончания интегрирования",endtime_angle_ball_for_step2)

argument1 = list(reversed(yprint_s_establ_angle_step1[0]))
argument2 = list(reversed(yprint_s_establ_angle_step1[1]))
argument3 = list(reversed(yprint_s_establ_angle_step1[2]))


argument1 = list((yprint_s_establ_angle_step1[0]))
argument2 = list((yprint_s_establ_angle_step1[1]))
argument3 = list((yprint_s_establ_angle_step1[2]))
#argument1 = list(reversed(yprint_s_establ_angle_step1.row(0)))
#argument2 = list(reversed(yprint_s_establ_angle_step1.row(1)))
#argument3 = list(reversed(yprint_s_establ_angle_step1.row(2)))

print('Число итераций',iterations_establ_angle_step1)
print('Время окончания', endtime_establ_angle_step1)
# print('Форма выходного вектора',xprint_s_bisect.shape)
# print('Форма выходного вектора',xprint_s_bisect.row(0).shape)
# print('Форма выходного вектора',xprint_s_bisect.row(1).shape)
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

#МЕТОД УСТАНОВЛЕНИЯ ШАГ 2
#endtime_angle_ball_for_step2 = 2.37349620145370
matrix_size = 2
k_angle_ball = 3
g = 9.81
A_Angleballaftereps = Matrix([k_angle_ball, g])  # смена Numpy матрицы на Sympy
ystart_Angleballaftereps = Matrix([[5.], [5.]])  # xx,VV            #смена Numpy матрицы на Sympy
dydx_Angleballaftereps = zeros(matrix_size, 1)

def dxdt_y_establstep2(A_Angleballaftereps, ystart_Angleballaftereps, b2x2):  # система уравнений для проверки поиска точки переключения

    # print('Shape of argument ',xstartswitchpoint.shape)
    # print('Shape of derivative ', dydx.shape)
    # print(type(xstartswitchpoint), 'Аргумент')
    # print(type(dydx[0, 0]), 'Производная')
    dydx_Angleballaftereps[0, 0] = ystart_Angleballaftereps[1, 0]
    # dydx[0, 0] = sum([dydx[0, 0], xstartswitchpoint[1, 0]])  # put args in list
    dydx_Angleballaftereps[1, 0] = (-1 * A_Angleballaftereps[0, 0] * (ystart_Angleballaftereps[1, 0]) - 9.81)  ##Vy' = -k*Vy-g

    # print(dydx)
    return dydx_Angleballaftereps





J1_switchball_angle_ball_step2 = sympy.Function('J')(x, y)

#f1_swpx_angle_ball_step2 = 'y'                   #  x' = y
#f2_swpy_angle_ball_step2 = '(-' + str(A_Angleballbeforeeps[0, 0]) + "*y-9.81)"  # y'= (-k*y-9.81)
f1_swpx_angle_ball_step2 = '0'
f1_swpy_angle_ball_step2 = '1'
f2_swpx_angle_ball_step2 = '0'
f2_swpy_angle_ball_step2 = '3'

F3_switchball_angle_step2 = Matrix([[f1_swpx_angle_ball_step2, f1_swpy_angle_ball_step2],
                                    [f2_swpx_angle_ball_step2, f2_swpy_angle_ball_step2]])
# print("Якобиан-1 переключения ", F2.jacobian([x, y]))

'''
allo = \
    Matrix([
        [0, 1],
        [0, -A_Angleballbeforeeps[0, 0]],
    ])
# print("Якобиан-2 переключения ", F2.jacobian([x, y]).subs([(x, 5), (y, 5)]))
'''

JacobianF_falling_angle_ball_step2 = Matrix(F3_switchball_angle_step1.subs([(x,5),(y,5)]))
print('Якоби 2x2 Мяч под углом ', JacobianF_falling_angle_ball_step2)
jaceiglist = list(JacobianF_falling_angle_ball_step2.eigenvals())
print('Собственные числа,2x2 Мяч под углом', jaceiglist)
#print('Шаг равен Мяч под углом', 2 / (reduce(lambda x, y: abs(x) if abs(x) > abs(y) else abs(y), jaceiglist)))



def hstabilitygetting_angle_ball_step2(MatrixForYacobian, values):
    JacobianF_falling_angle_ball_step2 = Matrix(
    MatrixForYacobian.subs([(x, values[0]), (y, values[1])]))
    # print('Якоби 3x3', JacobianF_falling_ball_step1)
    jaceiglist = list(JacobianF_falling_angle_ball_step2.eigenvals())
    # print('Собственные числа,3x3',jaceiglist)
    #jaceiglist = list(JacobianF_falling_angle_ball_step1.eigenvals())

    eigenlist = []

    for i in range(len(jaceiglist)):
        eq = sympy.S(str(jaceiglist[i]))
        eq = eq.subs(sympy.Function('arctg'), sympy.atan)
        eq = eq.subs(sympy.Function('sqrt'), math.sqrt)
        eq = float((eq).evalf())
        eigenlist.append(eq)

    #print("Собственные числа", eigenlist)
    return  2 / (reduce(lambda x, y: abs(x) if abs(x) > abs(y) else abs(y), eigenlist))


# def hstabilitygetting_falling_ball_step1(MatrixForYacobian, values):
# JacobianF = Matrix(MatrixForYacobian.jacobian([x, y, z]).subs([(x, values[0]), (y, values[1]), (z, values[2])]))
# jaceiglist = list(JacobianF.eigenvals())
# return 2 / (reduce(lambda x, y, z: d=abs(a) if abs(a), jaceiglist))

jaceiglist_falling_angle_ball_step2 = list(JacobianF_falling_angle_ball_step2.eigenvals())
# hbegin_falling_ball_step1 = 2 / (reduce(lambda x, y, z : abs(x) if (abs(x) > abs(y) and abs(x) > abs(z))
#                        else abs(y) if (abs(y) > abs(z)) else z , jaceiglist_falling_ball_step1))


hbegin_falling_ball_angle_ball_step2 = hstabilitygetting_angle_ball_step2(F3_switchball_angle_step2, [5.,5.])/4000
print("Начальный шаг по устойчивости для первого шага мяча 2x2 Мяч под углом", hbegin_falling_ball_angle_ball_step2)



#'''##Требует интегрирования МЕТОД УСТАНОВЛЕНИЯ МЯЧ ПОД УГЛОМ ШАГ 2
iterations_establ_angle_step2,endtime_establ_angle_step2, yprint_s_establ_angle_step2 = rkf2stepcontrol(matrix_size,
                                                                                    dydx_Angleballaftereps
                                                                                    , A_Angleballaftereps,
                                                                                    ystart_Angleballaftereps,
                                                                                    b2x2, dxdt_y_establstep2,
                                                                                    hbegin_falling_ball_angle_ball_step2,
                                                                                    #начальный шаг по устойчивости огромный
                                                                                    # НА СТАЦИОНАР,КАК НА КАРТИНКЕ
                                                                                    # hbegin_falling_ball_step1_alt,
                                                                                    0
            #                                , 20, 0.1,predicate_function_establ_step1, F3, hstabilitygetting) #Старое
                                                                                    , endtime_angle_ball_for_step2,
                                                                                    # ВРЕМЯ, ЗА КОТОРОЕ ВЫЙДЕМ НА СТАЦИОНАР
                                                                                    0.0000001,  # ТРЕБУЕМАЯ ТОЧНОСТЬ
                                                                                    # 0.000001, ДОЛГО
                                                                                    # 0.01, ЗАКАНЧИВАЕТ В 3.63, НЕ В НУЛЕ
                                                                                    # 0.9,
                                                                                    predicate_function_establ,
                                                                                    F3_switchball_angle_step2,
                                                                                    hstabilitygetting_angle_ball_step2,0.001)




#xprint_s_establ_angle_step2(0)
#yprint_s_establ_angle_step2.col_del(0)
yprint_s_establ_angle_step2 = numpy.delete(yprint_s_establ_angle_step2, 0, -1)
argument1 = list((yprint_s_establ_angle_step2[0]))
argument2 = list((yprint_s_establ_angle_step2[1]))
#argument1 = list(reversed(yprint_s_establ_angle_step2.row(0)))
#argument2 = list(reversed(yprint_s_establ_angle_step2.row(1)))
#argument3 = list(reversed(xprint_s_establ_angle_step2.row(2)))

print('Число итераций',iterations_establ_angle_step2)
print('Время окончания', endtime_establ_angle_step2)
# print('Форма выходного вектора',xprint_s_bisect.shape)
# print('Форма выходного вектора',xprint_s_bisect.row(0).shape)
# print('Форма выходного вектора',xprint_s_bisect.row(1).shape)
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


# print("Параметры",xstartlinearelersecondvarstep)
hw_pivot_angle_ball = sympy.zeros(1,1)

print("Параметры",A_Angleballaftereps,
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
    # d = -Alineareiler[1,0]/Alineareiler[0,0]
    start_time = datetime.now()
    print("Я в эйлере мяч под углом")
    Yold = ystartlinearelersecondvarstep[0, 0]  # xold = x
    VYold = ystartlinearelersecondvarstep[1, 0]  # vxold = vx
    #ystartlinearelersecondvarstep[2, 0] = ystartlinearelersecondvarstep[0, 0]
    endtime_eilermethodvariablestep = 0
    # xstartlineareilersecondvarstep[2, 0] = xstartlineareilersecondvarstep[0, 0]  # g = x
    # xstartlineareilersecondvarstep[2, 0] = xstartlineareilersecondvarstep[0, 0]  # g = x
    # print('g = ', xstartlinearelersecond[2, 0])
    h_for_output = []
    event_vect = []
    ystartlineareler_discrete_print_variablstep = zeros(3, 1)
    ##and (hweilervarstep_angle_ball > 0)
    while (ystartlinearelersecondvarstep[2, 0] > epsilon_eiler_second_angle_ball):  # g>eps
        print("Я внутри эйлера мяч под углом")
        print("ВРЕМЯ", endtime_eilermethodvariablestep)
        ystartlinearelersecondvarstep[2, 0] = ystartlinearelersecondvarstep[0, 0]  # g = x



        hpeilersecondvarstep_angle_ball = (gammaeilersecond_angle_ball - 1) * ystartlinearelersecondvarstep[2, 0] \
                               / (-3 * VYold - 9.81 + VYold)  # hp = (gamma-1)*g/(-omega**2*xold+Vxold)

        #event_vector = ystartlinearelersecondvarstep[2, 0] / (-3 * VYold - 9.81 + VYold)  -0.05
        event_vector = ystartlinearelersecondvarstep[2, 0] / (-3 * VYold - 9.81 + VYold)
        print("Шаги p и h",hpeilersecondvarstep_angle_ball, heilersecond_angle_ball)
        hweilersecondvarstep_angle_ball = min(hpeilersecondvarstep_angle_ball, heilersecond_angle_ball)  # hw = hp
        print("Шаг после", hweilersecondvarstep_angle_ball)
        endtime_eilermethodvariablestep = endtime_eilermethodvariablestep + hweilersecondvarstep_angle_ball

        # print('Шаг после', hweilervarstep)
        Fi = Matrix([VYold, -3  * VYold - 9.81])  # Fi = [Vxold;-omega**2*Xold]
        ystartlinearelersecondvarstep[0, 0] = Yold + hweilersecondvarstep_angle_ball * Fi[0, 0]  # x = xold+hw*Fi[1]
        ystartlinearelersecondvarstep[1, 0] = VYold + hweilersecondvarstep_angle_ball * Fi[1, 0]  # vx = vxold+hw*Fi[2]
        Yold = ystartlinearelersecondvarstep[0, 0]  # xold = x
        VYold = ystartlinearelersecondvarstep[1, 0]  # vxold = vx
        # print("Вектор решения после", xstartlineareilersecondvarstep)
        ystartlineareler_discrete_print_variablstep = \
            ystartlineareler_discrete_print_variablstep.col_insert(1, Matrix([ystartlinearelersecondvarstep]))
        print("Аргумент", ystartlinearelersecondvarstep)

        h_for_output.append(hweilersecondvarstep_angle_ball)
        event_vect.append(event_vector)
        hweeee = hweeee.col_insert(1, Matrix([hweilersecondvarstep_angle_ball]))
        print("Вектор шага",hweeee)
        # if (xstartlineareilersecondvarstep[2,0]<=epsilon_eiler_second)or(hweilersecondvarstep<eps):   #if g<=eps
        if (ystartlinearelersecondvarstep[2, 0] <= epsilon_eiler_second_angle_ball):  # if g<=eps

            end_time = datetime.now()
            print('Duration: {}'.format(end_time - start_time))
            return ystartlineareler_discrete_print_variablstep, endtime_eilermethodvariablestep,h_for_output,event_vect
            print(type(ystartlineareler_discrete_print_variablstep), "Тип матрицы перед возвращением")


#print(ystartlineareler_discrete_print_variablstep)

#'''#МЕТОД ЭЙЛЕРА
solutionlineareiler_angle_ball, endtime_eiler_angle_ball,hw_pivot_angle_ball,event_vect = eilermethod_angle_ball(A_Angleballaftereps,
                                                                                          ystartlinearelersecondvarstep,
                                                                                          gammaeilersecond_angle_ball,
                                                                                          hweilersecondvarstep_angle_ball,
                                                                                          hpeilersecondvarstep_angle_ball,
                                                                                          heilersecond_angle_ball,
                                                                                          epsilon_eiler_second_angle_ball,
                                                                                          N_eiler_linear_second, hw_pivot_angle_ball)

print(solutionlineareiler_angle_ball.shape,"Форма")

solutionlineareiler_angle_ball.col_del(0)
#del event_vect[-1]
# print(eilermethodvariablestep(Alineareilersecond,xstartlinearelersecond,gammaeilersecond,
#                    hweilersecondvarstep,hpeilersecondvarstep,heilersecond,epsilon_eiler_second,N_eiler_linear_second))


# ВЫВОД ГРАФИКА ПРИ МЕТОДЕ ЭЙЛЕРА С УПРАВЛЯЕМЫМ ШАГОМ   #X,VX
# print('Выходной вектор аргументов',solutionlineareilersecondconststep)
argument1 = list(reversed(solutionlineareiler_angle_ball.row(0)))
argument2 = list(reversed(solutionlineareiler_angle_ball.row(1)))
argument3 = list(reversed(solutionlineareiler_angle_ball.row(2)))
argument4 = list(reversed(event_vect))
# print('Число итераций',iterationseilerconststep)
print('Время окончания', endtime_eiler_angle_ball)
# print('Форма выходного вектора',xprint_s_bisect.shape)
# print('Форма выходного вектора',xprint_s_bisect.row(0).shape)
# print('Форма выходного вектора',xprint_s_bisect.row(1).shape)
t = linspace(0, float(endtime_eiler_angle_ball), solutionlineareiler_angle_ball.shape[1])


#plot(t, argument1, '-o', t, argument2, '-o', linewidth=2)

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
    #return xstart[0, 0] < 0
    #return xstart[0, 0] < 0+0.0001
    #return xstart[0,0] < 0 + 0.0001
    return float(xstart[0]) < 0 + eps

#print("Предикат в 2.36429961675833",predicate_function_linear_xsecond([[0.0231390146588621], [-3.26312775428630]]))

#ystart_Angleballaftereps = Matrix([[0.05], [-3.26312775428630]])

#'''#Метод Эспосито
iterations_rkeiler_angle_step2,endtime_rkeiler_angle_step2, yprint_s_rkeiler_angle_step2 = rkf2stepcontrol(matrix_size,
                                                                                    dydx_Angleballaftereps
                                                                                    , A_Angleballaftereps,
                                                                                    ystart_Angleballaftereps,
                                                                                    b2x2, dxdt_y_establstep2,
                                                                                       hweilerbasedrungekutta,
                                                                                    #начальный шаг по устойчивости огромный
                                                                                    # НА СТАЦИОНАР,КАК НА КАРТИНКЕ
                                                                                    # hbegin_falling_ball_step1_alt,
                                                                                    #0
                                                                                    0
            #                                , 20, 0.1,predicate_function_establ_step1, F3, hstabilitygetting) #Старое
                                                                                    , 10,
                                                                                    # ВРЕМЯ, ЗА КОТОРОЕ ВЫЙДЕМ НА СТАЦИОНАР
                                                                                    epsilon_eiler_second_angle_ball,  # ТРЕБУЕМАЯ ТОЧНОСТЬ
                                                                                    # 0.000001, ДОЛГО
                                                                                    # 0.01, ЗАКАНЧИВАЕТ В 3.63, НЕ В НУЛЕ
                                                                                    # 0.9,
                                                                                    predicate_function_linear_xsecond,
                                                                                    F3_switchball_angle_step2,
                                                                                    hstabilitygetting_angle_ball_step2,hweilerbasedrungekutta)




#xprint_s_establ_angle_step2(0)
#yprint_s_rkeiler_angle_step2.col_del(0)
yprint_s_rkeiler_angle_step2 = numpy.delete(yprint_s_rkeiler_angle_step2, 0, -1)
argument1 = list((yprint_s_rkeiler_angle_step2[0]))
argument2 = list((yprint_s_rkeiler_angle_step2[1]))
#argument1 = list(reversed(yprint_s_rkeiler_angle_step2.row(0)))
#argument2 = list(reversed(yprint_s_rkeiler_angle_step2.row(1)))
#argument3 = list(reversed(xprint_s_establ_angle_step2.row(2)))

print('Число итераций',iterations_rkeiler_angle_step2)
print('Время окончания', endtime_rkeiler_angle_step2)
# print('Форма выходного вектора',xprint_s_bisect.shape)
# print('Форма выходного вектора',xprint_s_bisect.row(0).shape)
# print('Форма выходного вектора',xprint_s_bisect.row(1).shape)
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
#'''#Метод Эспосито