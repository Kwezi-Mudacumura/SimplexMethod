import numpy as np

""" Exercise 1 """
def makeTableau(c, A, b):
    n = len(A)

    # turn into a maximisation problem
    c = c*(-1)

    # Create an identity matrix for the slack variables
    I = np.eye(n)
    table = np.hstack((A, I))

    # make b into a column
    b = b.reshape((len(b), 1))
    table = np.hstack((table, b))

    z = np.zeros(n+1)
        
    #Add slack variables to objective function
    new = np.hstack((c, z))
    table = np.vstack((new, table))
    
    c_size = len(c)
    I_size = len(I)
    
    basis = np.arange(c_size, c_size + I_size)

    return table, basis

# A = np.array([[3,-4,1], [4,5,-4]], float)
# c = np.array([0,2,5], float)
# b = np.array([49, 8], float)

# tableau, basis = makeTableau(c, A, b)
# print("Exercise 1: \n", tableau, "\n", basis)

def size_x(tableau):
    
    return tableau.shape[1] - 1
    
    #return len(tableau[0])-1

""" Exercise 2 """
def readSolution(tableau, basis):
    # Z value is the last value in the objective row
    
    # last value in each row
    row_size = size_x(tableau)
    
    # fist row, last column
    z = tableau[0, row_size]

    #x = np.zeros(len(tableau[0])-1)
    x = np.zeros(row_size)
    
    # Basis values is the last value in the respective row
    
    x[basis] = tableau[1:, -1]
    
    #zeros = np.zeros(len(tableau[1:, -1]))
    #np.copyto(x[basis], tableau[1:, -1])
    
#     x = np.zeros(len(tableau[0]) - 1)
#     for i, basis_index in enumerate(basis):
#         x[basis_index] = tableau[i + 1, -1]

    return x, z

# x, z = readSolution(tableau, basis)
# print("Exercise 2: \n", x, z)

def minimum(tableau):
    return np.argmin(tableau[0, :-1])

""" Exercise 3 """
def getPivotColumn(tableau):
    x = minimum(tableau)
    if(tableau[0][x] >= 0):
        x = -1
    return x

    # #Find the index of the minimum value in the first row
#    min_index = np.argmin(tableau[0, :-1])
    
#     #Check if the minimum value is non-negative
#    if tableau[0, min_index] >= 0:
#        min_index = -1
    
#    return min_index
    

# pivotCol = getPivotColumn(tableau)
# print("Exercise 3: \n", pivotCol)


""" Exercise 4 """
def findMin(a, b):
    # return smallest entry of a/b such that b > 0. Otherwise, return -1
    # a and b are numpy array
    index = -1
    minFound = np.inf
    for i in range(len(b)):
        # b to be the b vector
        if (a[i] > 0) and (b[i]/a[i] < minFound):
            minFound = b[i]/a[i]
            index = i
    return index

#     value = a > 0
#     index = np.argmin(np.where(value, b / a, np.inf))
#     return index
    

def getPivotRow(tableau, pivotCol):
    return findMin(tableau[1:, pivotCol], tableau[1:, -1])

# pivotRow = getPivotRow(tableau, pivotCol)
# print("Exercise 4 \n", pivotRow)


""" Exercise 5 """
def getPivotElement(tableau):
    pivotCol = getPivotColumn(tableau)
    pivotRow = getPivotRow(tableau, pivotCol)
    return np.array([pivotRow, pivotCol])

# pivotElement = getPivotElement(tableau)
# print("Exercise 5: \n", pivotElement)


""" Exercise 6 """
def updateBasis(basis, pivotElement):
    basis[pivotElement[0]] = pivotElement[1]
    return basis

# newBasis = updateBasis(basis, pivotElement)
# print("Exercise 6: \n" , newBasis)

# tableau, basis = makeTableau(c, A, b)

# def hold(tableauPivotRow,pivotCol):
#     return tableauPivotRow[pivotCol]

""" Exercise 7 """
def makePivotElement1(tableauPivotRow, pivotCol):
    M = tableauPivotRow[pivotCol]
    if M != 1:
        tableauPivotRow[:] = tableauPivotRow[:]/M
    return tableauPivotRow

def updateTableauRow(tableauPivotRow, tableauRow, pivotCol):
    M = tableauRow[pivotCol]
    if M != 0:
        tableauRow[:] = tableauRow[:] - M*tableauPivotRow[:]
    return tableauRow

# pivotRow = pivotElement[0] + 1
# pivotCol = pivotElement[1]

# tableauPivotRow = tableau[pivotRow, :]
# tableauRow = tableau[2, :]

# print("Exercise 7: \n", updateTableauRow(tableauPivotRow, tableauRow, pivotCol))

""" Exercise 8 """
# Example 1
# A = np.array([[3,-4,1], [4,5,-4]], float)
# c = np.array([0,2,5], float)
# b = np.array([49, 8], float)

# tableau, basis = makeTableau(c, A, b)

#Example 2

# tableau = np.array([[-6, 9, 2, 0, 0, 0, 0, 0],
#                     [-4, -7, 0, 1, 0, 0, 0, 5],
#                     [-1, -8, -3, 0, 1, 0, 0, 4],
#                     [-10, -3, 9, 0, 0, 1, 0, 7],
#                     [10, -6, 9, 0, 0, 0, 1, 6]], float)

# basis = np.array([3,4,5,6])
# pivotElement = getPivotElement(tableau)

def updateTableau(tableau, basis, pivotElement):
    basis = updateBasis(basis, pivotElement)
    pivotRow = pivotElement[0] + 1
    pivotCol = pivotElement[1]
    tableauPivotRow = tableau[pivotRow, :]
    tableau[pivotRow, :] = makePivotElement1(tableauPivotRow, pivotCol)
    # tableau[pivotRow, :] = tableau[pivotRow, :]/tableauPivotRow[pivotCol]

    for i in range(tableau.shape[0]):
        if i != pivotRow:
            tableau[i, :] = updateTableauRow(tableauPivotRow, tableau[i, :], pivotCol)

    return tableau, basis

# tableau, basis = updateTableau(tableau, basis, pivotElement)
# print("Exercise 8: \n", tableau, basis)

""" Exercise 9 """
def pivotPivotElementMultipleSolution(tableau, basis):
    tableau[0, basis] = 1
    pivotCol = np.argmin(tableau[0, :-1])

    if tableau[0, pivotCol] > 0:
        pivotElement = np.array([-1, -1])
    else:
        pivotRow = getPivotRow(tableau, pivotCol)
        if pivotRow == -1:
            pivotElement = np.array([-1, -1])
        else:
            pivotElement = np.array([pivotRow, pivotCol])

    return pivotElement

def simplexMethod(c, A, b):
    iteration = 0
    tableau, basis = makeTableau(c, A, b)
    pivotElement = getPivotElement(tableau)

    while pivotElement[1] != -1:
        iteration += 1
        if pivotElement[0] == -1:
            return 0
        else:
            tableau, basis = updateTableau(tableau, basis, pivotElement)
            pivotElement = getPivotElement(tableau)

    pivotElement = pivotPivotElementMultipleSolution(tableau, basis)
    solution, z = readSolution(tableau, basis)

    if pivotElement[0] != -1:
        tableau, basis = updateTableau(tableau, basis, pivotElement)
        pivotElement = getPivotElement(tableau)
        solution2, z = readSolution(tableau, basis)
        return solution[0:len(c)], solution2[0:len(c)], z
    else:
        return solution[0:len(c)], z

# c = np.array([-1, 2], float)
# A = np.array([[1, 1], [1, -1]], float)
# b = np.array([1,3], float)
# print("Exercise 9: \n", simplexMethod(c, A, b))

    
# c = np.array([36, 30, -3,  -4], float)
# A = np.array([[1, 1, -1, 0], [6, 5, 0, -1]], float)
# b = np.array([5, 10], float)
# print("Exercise 9: \n", simplexMethod(c, A, b))

# c = np.array([40, 40, 90], float)
# A = np.array([[-6, 6, -2], [4, 4, 9], [7, -10, -10], [-3, 2, -1]], float)
# b = np.array([7, 7, 0, 2], float)
# print("Exercise 9: \n", simplexMethod(c, A, b))

# c = np.array([1, -3, -2], float)
# A = np.array([[3, -1, 2], [-2, 4, 0], [-4, 3, 8]], float)
# b = np.array([7, 12, 10], float)
# print("Exercise 9: \n", simplexMethod(-c, A, b))




