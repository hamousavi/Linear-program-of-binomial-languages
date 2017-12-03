#########################################################
#########################################################

# This program uses CVXPY. To install you must visit
# http://www.cvxpy.org/en/latest/install/index.html

# To run, enter the following line in the command line
# $ python relaxation.py 10 3
# The first number is n and the second number is k. 
# This will solve the linear program P'(n,k).

# This program finds an optimal solution of P'(n,k). 
# Default n and k are defined in the following lines. 
n = 10
k = 3

#########################################################
#########################################################
from itertools import permutations
import cvxpy as cvx 
import numpy as np
from collections import OrderedDict
from operator import itemgetter
from scipy.special import binom
import math
import itertools

if __name__ == '__main__':
    from sys import argv
    if len(argv) == 3:
    	n = int(argv[1])
    	k = int(argv[2])

print 'Creating problem P\'('+str(n)+','+str(k)+') ... '

# Calculating C_c(n,k)
C = list()
for l in range(0,k+1):
	for m in range(max(1,l),n+1):
		C.append((m,l))

# Calculating C_c(n,k)
C_c = list()
for (n1,k1) in C:
	for (n2,k2) in C:
		if n1+n2 <= n and k1+k2 <= k: 
			C_c.append((n1,k1,n2,k2))

# Calculating B(m,l) for all (m,l) in C(n,k)
B = dict()
INDEX = dict()
C_0 = list()
i = 0
for (m,l) in C:
	if l == 0:
		B[(m,l)] = ['0'*m]
	elif m == l:
		B[(m,l)] = ['1'*m]
	else:
		B[(m,l)] = [s+'0' for s in B[(m-1,l)]] + [s+'1' for s in B[(m-1,l-1)]]
	for s in B[(m,l)]:
		C_0.append(s)
		INDEX[s] = i
		i = i + 1

number_of_variables = i

# Defining x which is the variable in our optimization problem
x = cvx.Variable(number_of_variables)

# Defining objective function b^T x
b = np.zeros((number_of_variables,), dtype=np.int)
for s in B[(n,k)]:
	b[INDEX[s]] = 1

objective_function = cvx.Maximize(b.T * x)

# J is the vector for which J_s = |s| for all s \in C_0(n,k)
J = np.zeros((number_of_variables,), dtype=np.int)
for s in C_0:
	J[INDEX[s]] = len(s)

positivity_constraints = [x >= 0]
boundedness_constraints = [x <= J]

subadditivity_constraints = []
for (n1,k1,n2,k2) in C_c:
	I_n1_k1 = np.zeros((number_of_variables,), dtype=np.int)
	I_n2_k2 = np.zeros((number_of_variables,), dtype=np.int)
	# I_n1_k1_n2_k2 is the vector such that I_n1_k1_n2_k2(s) is 1 whenever s \in B(n1,k1)B(n2,k2)
	# and zero everywhere else	
	I_n1_k1_n2_k2 = np.zeros((number_of_variables,), dtype=np.int)
	for s in B[(n1,k1)]:
		I_n1_k1[INDEX[s]] = 1
	for s in B[(n2,k2)]:
		I_n2_k2[INDEX[s]] = 1
	for s in B[(n1,k1)]:
		for t in B[(n2,k2)]:
			I_n1_k1_n2_k2[INDEX[s+t]] = 1
		
	subadditivity_constraints.append((I_n1_k1_n2_k2-I_n1_k1-I_n2_k2).T * x<= 0)

constraints =  positivity_constraints + boundedness_constraints + subadditivity_constraints
prob = cvx.Problem(objective_function, constraints)
	
print 'Solving P\'('+str(n)+','+str(k)+') ...'
prob.solve()

print 'We are done with the status \"', prob.status,'\"'


print '----------------------------------------------------------------------------------------'
print '----------------------------------------------------------------------------------------'
print '----------------------------------------------------------------------------------------'
print '----------------------------------------------------------------------------------------'
print '----------------------------------------------------------------------------------------'

print 'We found the following optimal solution'
for (m,l) in C:
	for s in B[(m,l)]:
		print ''.join(s),x[INDEX[s]].value

print '----------------------------------------------------------------------------------------'
print '----------------------------------------------------------------------------------------'
print '----------------------------------------------------------------------------------------'
print '----------------------------------------------------------------------------------------'
print '----------------------------------------------------------------------------------------'
print 'The quantity ' + 'opt(P\'('+str(n)+','+str(k)+')) is ', prob.value

print ''
def Ellul(m,l):
	floor = int(math.floor(m/2.0))
	ceil = m - floor
	if l == 0:
		return m
	if m == l:
		return m
	if l > floor:
		return Ellul(m,m-l)
	res = 0
	for i in range(0,l+1):
		res = res + Ellul(floor,i)+Ellul(ceil,l-i)
	return res

print '... and |R_' + str(n) + '_' + str(k) + '| is ', str(Ellul(n,k))+','
print ''
print 'where R_' + str(n) + '_' + str(k) + ' is the regular expression from the Ellul et al. paper.'
