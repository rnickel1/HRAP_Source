import numpy as np
from numpy import genfromtxt

# Simple util for converting saved CSVs to bundled npz
# Run in the same folder as the exported CSVs, with filenames given below
# To generate CSVs from MAT files, use csvwrite in Matlab
OUTFILE = 'Paraffin.npz'
PROP_NAME = 'Paraffin'

k = genfromtxt('k.csv', delimiter=',')
M = genfromtxt('M.csv', delimiter=',')
OF = genfromtxt('OF.csv', delimiter=',')
Pc = genfromtxt('Pc.csv', delimiter=',')
reg_coeff = genfromtxt('reg_coeff.csv', delimiter=',')
rho = genfromtxt('rho.csv', delimiter=',')
T = genfromtxt('T.csv', delimiter=',')

np.savez(OUTFILE, metadata=[PROP_NAME, rho], OF=OF, Pc=Pc, k=k, M=M, T=T, regression_coeff=reg_coeff)

