import numpy as np

# Question 1

def function (t: float, y: float):
    return (t - (y**2))


def eulers(t_0,y_0,t_f,n):
    h = (t_f - t_0) / n

    for i in range(0,n):
        y_n = y_0 + (h * (function(t_0,y_0)))
        y_0 = y_n
        t_0 = t_0 + h 
    print (y_n)

eulers(0,1,2,10)
# Question 2

def func(t, y):
    return (t - y**2)

def runge_kutta(t_0, t_f, n,y_0):
    h = (t_f - t_0) / n
    y = y_0
    t = t_0
    for another_unused_variable in range(n):
        k_1 = h * func(t, y)
        k_2 = h * func((t + (h / 2)), (y + (k_1 / 2)))
        k_3 = h * func((t + (h / 2)), (y + (k_2 / 2)))
        k_4 = h * func((t + h), (y + k_3))

        y = y + (1 / 6) * (k_1 + (2 * k_2) + (2 * k_3) + k_4)

        t = t + h

    print(y, "\n")


runge_kutta(0,2,10,1)
# Question 3
def gauss_elimination(gauss_matrix):
    size = gauss_matrix.shape[0]

    for i in range(size):
        pivot = i
        while gauss_matrix[pivot, i] == 0:
            pivot += 1
   
        gauss_matrix[[i, pivot]] = gauss_matrix[[pivot, i]]

        for j in range(i + 1, size):
            factor = gauss_matrix[j, i] / gauss_matrix[i, i]
            gauss_matrix[j, i:] = gauss_matrix[j, i:] - factor * gauss_matrix[i, i:]

    inputs = np.zeros(size)

    for i in range(size - 1, -1, -1):
        inputs[i] = (gauss_matrix[i, -1] - np.dot(gauss_matrix[i, i: -1], inputs[i:])) / gauss_matrix[i, i]
   
    final_answer = np.array([int(inputs[0]), int(inputs[1]), int(inputs[2])])
    print(final_answer, "\n")

gauss_matrix = np.array([[2, -1, 1, 6], [1, 3, 1, 0], [-1, 5, 4, -3]])
gauss_elimination(gauss_matrix)
# Question 4

def lu_factorization(lu_matrix):
    size = lu_matrix.shape[0]

    l = np.eye(size)
    u = np.zeros_like(lu_matrix)

    for i in range(size):
        for j in range(i, size):
            u[i, j] = (lu_matrix[i, j] - np.dot(l[i, :i], u[:i, j]))
   
        for j in range(i + 1, size):
            l[j, i] = (lu_matrix[j, i] - np.dot(l[j, :i], u[:i, i])) / u[i, i]
   
    determinant = np.linalg.det(lu_matrix)

    print(determinant, "\n")
    print(l, "\n")
    print(u,"\n")

lu_matrix = np.array([[1, 1, 0, 3], [2, 1, -1, 1], [3, -1, -1, 2], [-1, 2, 3, -1]], dtype = np.double)
lu_factorization(lu_matrix)
# Question 5

def diagonally_dominant(dd_matrix):
    n = 5
    for i in range(0, n):
        total = 0
        for j in range(0, n):
            total = total + abs(dd_matrix[i][j])
       
        total = total - abs(dd_matrix[i][i])
   
    if abs(dd_matrix[i][i]) < total:
        print("False\n")
    else:
        print("True\n")

dd_matrix = np.array([[9, 0, 5, 2, 1], [3, 9, 1, 2, 1], [0, 1, 7, 2, 3], [4, 2, 3, 12, 2], [3, 2, 4, 0, 8]])
diagonally_dominant(dd_matrix)
# Question 6

def positive_definite(pd_matrix):
    eigenvalues = np.linalg.eigvals(pd_matrix)

    if np.all(eigenvalues > 0):
        print("True\n")
    else:
        print("False\n")
        
pd_matrix = np.matrix([[2, 2, 1], [2, 3, 0], [1, 0, 2]])
positive_definite(pd_matrix)