#!/usr/bin/env python3
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

i = np.complex(0,1)

# Ground State Qubits 0-4
Q0 = np.matrix([[1],[0]])
Q1 = np.matrix([[1],[0]]) 
Q2 = np.matrix([[1],[0]])
Q3 = np.matrix([[1],[0]])
Q4 = np.matrix([[1],[0]])

# [X], [Y], [Z] Pauli Gates
X = np.matrix([[0,1],[1,0]])		# |0> -> X|0> = |1>
Y = np.matrix([[0,-i],[i,0]])
Z = np.matrix([[1,0],[0,-1]])		# flips phase (|1> to -|1>)

# Superposition Gates
H = 1/np.sqrt(2) * np.matrix([[1,1],[1,-1]])	# Hadamard Gate
S = np.matrix([[1,0],[0,i]])					# Phase Gate
Sd = np.matrix([[1,0],[0,-i]])

# CNOT Gate -- Control A, target B; on IBM5Q target needs to be Q2
CNOT = np.matrix([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]])

# non-Clifford Gates
T = np.matrix([[1,0],[0, np.e**(i*np.pi/4)]])
Td = np.matrix([[1,0],[0, np.e**(-i*np.pi/4)]])

# Z = standard basis
# 	|+> = (1/sqrt(2))*(|0> + |1>)	measure in diagonal basis (X) to
#	|-> = (1/sqrt(2))*(|0> - |1>)	differentiate these states
# Y = circular basis
#	|cw> = (1/sqrt(2))*(|0> + i|1>)
#	|ccw> = (1/sqrt(2))*(|0> - i|1>)

def calcBlochCoord(state):
	phi = np.angle(state[1,0]) - np.angle(state[0,0])
	theta = 2*np.arccos(abs(state[0,0]))
	print('phi: ' + repr(phi) + ', theta: ' + repr(theta))
	vx = np.sin(theta)*np.cos(phi)
	vy = np.sin(theta)*np.sin(phi)
	vz = np.cos(theta)
	return vx,vy,vz

def stateToBinary(state):
    # zero state = [[1],[0]]
    # one state = [[0],[1]]
    stateOpts = ('0', '1')
    stateAB = [state[0,0], state[1,0]]
    statePos = np.arange(len(stateOpts))
    binfig, axB = plt.subplots()
    axB.bar(statePos, stateAB, 0.15, align='center', color='green', alpha=0.5)
    plt.xticks(statePos, stateOpts)
    axB.set_ylim([0,1])
    plt.tight_layout()
    binfig.show()

def resetQubits():
	Q0 = np.matrix([[1],[0]])
	Q1 = np.matrix([[1],[0]]) 
	Q2 = np.matrix([[1],[0]])
	Q3 = np.matrix([[1],[0]])
	Q4 = np.matrix([[1],[0]])

def applyCNOTGate(qubitA, qubitB):		# A is control, B is target
	qubitAB = np.kron(qubitA, qubitB)
	qubitAcnotB = CNOT*qubitAB
	return qubitAcnotB
	
def qbitsToBinary(Q0,Q1,Q2,Q3,Q4):
	stateOpts = ('00000','00001','00010','00011','00100','00101','00110','00111','01000','01001','01010','01011','01100','01101','01110','01111','10000','10001','10010','10011','10100','10101','10110','10111','11000','11001','11010','11011','11100','11101','11110','11111')
	statePos = np.arange(len(stateOpts))

	pos00000 = Q0[0,0]*Q1[0,0]*Q2[0,0]*Q3[0,0]*Q4[0,0]
	pos00001 = Q0[0,0]*Q1[0,0]*Q2[0,0]*Q3[0,0]*Q4[1,0]
	pos00010 = Q0[0,0]*Q1[0,0]*Q2[0,0]*Q3[1,0]*Q4[0,0]
	pos00011 = Q0[0,0]*Q1[0,0]*Q2[0,0]*Q3[1,0]*Q4[1,0]
	pos00100 = Q0[0,0]*Q1[0,0]*Q2[1,0]*Q3[0,0]*Q4[0,0]
	pos00101 = Q0[0,0]*Q1[0,0]*Q2[1,0]*Q3[0,0]*Q4[1,0]
	pos00110 = Q0[0,0]*Q1[0,0]*Q2[1,0]*Q3[1,0]*Q4[0,0]
	pos00111 = Q0[0,0]*Q1[0,0]*Q2[1,0]*Q3[1,0]*Q4[1,0]
	pos01000 = Q0[0,0]*Q1[1,0]*Q2[0,0]*Q3[0,0]*Q4[0,0]
	pos01001 = Q0[0,0]*Q1[1,0]*Q2[0,0]*Q3[0,0]*Q4[1,0]
	pos01010 = Q0[0,0]*Q1[1,0]*Q2[0,0]*Q3[1,0]*Q4[0,0]
	pos01011 = Q0[0,0]*Q1[1,0]*Q2[0,0]*Q3[1,0]*Q4[1,0]
	pos01100 = Q0[0,0]*Q1[1,0]*Q2[1,0]*Q3[0,0]*Q4[0,0]
	pos01101 = Q0[0,0]*Q1[1,0]*Q2[1,0]*Q3[0,0]*Q4[1,0]
	pos01110 = Q0[0,0]*Q1[1,0]*Q2[1,0]*Q3[1,0]*Q4[0,0]
	pos01111 = Q0[0,0]*Q1[1,0]*Q2[1,0]*Q3[1,0]*Q4[1,0]
	pos10000 = Q0[1,0]*Q1[0,0]*Q2[0,0]*Q3[0,0]*Q4[0,0]
	pos10001 = Q0[1,0]*Q1[0,0]*Q2[0,0]*Q3[0,0]*Q4[1,0]
	pos10010 = Q0[1,0]*Q1[0,0]*Q2[0,0]*Q3[1,0]*Q4[0,0]
	pos10011 = Q0[1,0]*Q1[0,0]*Q2[0,0]*Q3[1,0]*Q4[1,0]
	pos10100 = Q0[1,0]*Q1[0,0]*Q2[1,0]*Q3[0,0]*Q4[0,0]
	pos10101 = Q0[1,0]*Q1[0,0]*Q2[1,0]*Q3[0,0]*Q4[1,0]
	pos10110 = Q0[1,0]*Q1[0,0]*Q2[1,0]*Q3[1,0]*Q4[0,0]
	pos10111 = Q0[1,0]*Q1[0,0]*Q2[1,0]*Q3[1,0]*Q4[1,0]
	pos11000 = Q0[1,0]*Q1[1,0]*Q2[0,0]*Q3[0,0]*Q4[0,0]
	pos11001 = Q0[1,0]*Q1[1,0]*Q2[0,0]*Q3[0,0]*Q4[1,0]
	pos11010 = Q0[1,0]*Q1[1,0]*Q2[0,0]*Q3[1,0]*Q4[0,0]
	pos11011 = Q0[1,0]*Q1[1,0]*Q2[0,0]*Q3[1,0]*Q4[1,0]
	pos11100 = Q0[1,0]*Q1[1,0]*Q2[1,0]*Q3[0,0]*Q4[0,0]
	pos11101 = Q0[1,0]*Q1[1,0]*Q2[1,0]*Q3[0,0]*Q4[1,0]
	pos11110 = Q0[1,0]*Q1[1,0]*Q2[1,0]*Q3[1,0]*Q4[0,0]
	pos11111 = Q0[1,0]*Q1[1,0]*Q2[1,0]*Q3[1,0]*Q4[1,0]
	
	allPositions = [pos00000, pos00001, pos00010, pos00011, pos00100, pos00101, pos00110, pos00111, pos01000, pos01001, pos01010, pos01011, pos01100, pos01101, pos01110, pos01111, pos10000, pos10001, pos10010, pos10011, pos10100, pos10101, pos10110, pos10111, pos11000, pos11001, pos11010, pos11011, pos11100, pos11101, pos11110, pos11111]
	binfig = plt.figure()
	axC = binfig.add_subplot(111)
	axC.bar(statePos, allPositions, 0.9, align='center', color='green', alpha=0.5)
	plt.xticks(statePos, stateOpts)
	axC.set_ylim([0,1])
	axC.set_title('IBM Random Classical Circuit, [Q0, Q2, Q4]-[X]--[Z]=, [Q1, Q3]--[Z]=, Expected: 10101')
	plt.tight_layout()
	binfig.show()	

## -------------------------------------------------- ##
## ----------- Pauli X, Y, Z Measurements ----------- ##
## -------------------------------------------------- ##
## Pauli X            |0>-[X]--[M]=1
## Pauli Y            |0>-[Y]--[M]=1
## Pauli Z            |0>-[Z]--[M]=0
Q0_xyz = X*Q0
print('Pauli X: ' + repr(Q0_xyz))
Q1_xyz = (Y*Q1).imag
print('Pauli Y: ' + repr(Q1_xyz))
Q2_xyz = Z*Q2
print('Pauli Z: ' + repr(Q2_xyz))
stateToBinary(Q0_xyz)
stateToBinary(Q1_xyz)
stateToBinary(Q2_xyz)

## -------------------------------------------------- ##
## ------------ Superposition (H, S, Sd) ------------ ##
## -------------------------------------------------- ##
## Superposition (+)               |0>-[H]---------------[M]=|+>	0.5/0.5
## Superposition (-)               |0>-[H]-[Z]-----------[M]=|->	0.5/0.5
## Superposition (+) X Basis       |0>-[H]-[H]-----------[M]=|0>
## Superposition (-) X Basis       |0>-[H]-[Z]-[H]-------[M]=|1>
## Superposition (+i) Y Basis      |0>-[H]-[S]--[Sd]-[H]-[M]=|0>
## Superposition (-i) Y Basis      |0>-[H]-[Sd]-[Sd]-[H]-[M]=|1>
Q0_sp = Z*(H*Q0)
print('Supoerposition (-) Std/Z Basis: ' + repr(Q0_sp))
Q1_sp = H*(H*Q0)
print('Superposition (+) Diag/X Basis: ' + repr(Q1_sp))
Q2_sp = H*(Z*(H*Q0))
print('Superposition (-) Diag/X Basis: ' + repr(Q2_sp))
Q3_sp = H*(S*(Sd*(H*Q0)))
print('Superposition (+i) Circ/Y Basis: ' + repr(Q3_sp))
Q4_sp = H*(Sd*(Sd*(H*Q0)))
print('Superposition (-i) Circ/Y Basis: ' + repr(Q4_sp))
stateToBinary(Q0_sp)
stateToBinary(Q1_sp)
stateToBinary(Q2_sp)
stateToBinary(Q3_sp)
stateToBinary(Q4_sp)

## -------------------------------------------------- ##
## ------------ IBM Controlled NOT (CNOT) ----------- ##
## -------------------------------------------------- ##
Q0_control = Q0
Q1_target = Q1
Q0cnotC1 = applyCNOTGate(Q0_control, Q1_target)
# TODO separate out the two


## -------------------------------------------------- ##
## ----- Random Classical Circuit [Exp: 10101] ------ ##
## -------------------------------------------------- ##
Q0x = X*Q0
Q2x = X*Q2
Q4x = X*Q4
qbitsToBinary(Q0x, Q1, Q2x, Q3, Q4x)
#resetQubits()

## -------------------------------------------------- ##
## --- Cardinal States Circuit; Bloch Tomography ---- ##
## -------------------------------------------------- ##
Q0_1 = Q0                           ## Q0-----------------[B]=
Q1_1 = X*Q0                         ## Q1--[X]------------[B]=
Q2_1 = (S*(H*Q0))                   ## Q2--[H]-[S]--------[B]=
print('Q2_1: ' + repr(Q2_1))        
Q3_1 = (Sd*(H*Q0))                  ## Q3--[H]-[Sd]-------[B]=
print('Q3_1: ' + repr(Q3_1))
Q4_1 = (Z*(H*Q0))                   ## Q4--[H]-[Z]--------[B]=
blochxyz0 = calcBlochCoord(Q0_1)
blochxyz1 = calcBlochCoord(Q1_1)
blochxyz2 = calcBlochCoord(Q2_1)
blochxyz3 = calcBlochCoord(Q3_1)
blochxyz4 = calcBlochCoord(Q4_1)

# plot bloch sphere
fig = plt.figure()

ax0 = fig.add_subplot(231, projection='3d')
ax1 = fig.add_subplot(232, projection='3d', sharex=ax0, sharey=ax0, sharez=ax0)
ax2 = fig.add_subplot(233, projection='3d', sharex=ax0, sharey=ax0, sharez=ax0)
ax3 = fig.add_subplot(234, projection='3d', sharex=ax0, sharey=ax0, sharez=ax0)
ax4 = fig.add_subplot(235, projection='3d', sharex=ax0, sharey=ax0, sharez=ax0)

ax0.set_xticklabels([])
ax0.set_xticks([])
ax0.set_yticklabels([])
ax0.set_yticks([])
ax0.set_zticklabels([])
ax0.set_zticks([])
ax0.w_xaxis.line.set_color("white")
ax0.w_yaxis.line.set_color("white")
ax0.w_zaxis.line.set_color("white")
ax1.w_xaxis.line.set_color("white")
ax1.w_yaxis.line.set_color("white")
ax1.w_zaxis.line.set_color("white")
ax2.w_xaxis.line.set_color("white")
ax2.w_yaxis.line.set_color("white")
ax2.w_zaxis.line.set_color("white")
ax3.w_xaxis.line.set_color("white")
ax3.w_yaxis.line.set_color("white")
ax3.w_zaxis.line.set_color("white")
ax4.w_xaxis.line.set_color("white")
ax4.w_yaxis.line.set_color("white")
ax4.w_zaxis.line.set_color("white")

qp = np.linspace(0, 2 * np.pi, 40)
qt = np.linspace(0, np.pi, 60)
xaxis = 1 * np.outer(np.cos(qp), np.sin(qt))
yaxis = 1 * np.outer(np.sin(qp), np.sin(qt))
zaxis = 1 * np.outer(np.ones(np.size(qp)), np.cos(qt))

# wireframe & axis guide lines
ax0.plot_wireframe(xaxis,yaxis,zaxis,rstride=10,cstride=60,color='grey',alpha=0.5)
ax0.plot(np.sin(qp),np.cos(qp),0,color='k',linestyle='--',alpha=0.5)
ax0.plot([0,1],[0,0],[0,0],color='k',linestyle=':',alpha=0.5)
ax0.plot([0,0],[0,1],[0,0],color='k',linestyle=':',alpha=0.5)
ax0.plot([0,0],[0,0],[0,1],color='k',linestyle=':',alpha=0.5)
ax0.plot([0,-1],[0,0],[0,0],color='k',linestyle=':',alpha=0.5)
ax0.plot([0,0],[0,-1],[0,0],color='k',linestyle=':',alpha=0.5)
ax0.plot([0,0],[0,0],[0,-1],color='k',linestyle=':',alpha=0.5)
ax0.set_title('|0>--[B]=', fontsize=10)
ax1.plot_wireframe(xaxis,yaxis,zaxis,rstride=10,cstride=60,color='grey',alpha=0.5)
ax1.plot(np.sin(qp),np.cos(qp),0,color='k',linestyle='--',alpha=0.5)
ax1.plot([0,1],[0,0],[0,0],color='k',linestyle=':',alpha=0.5)
ax1.plot([0,0],[0,1],[0,0],color='k',linestyle=':',alpha=0.5)
ax1.plot([0,0],[0,0],[0,1],color='k',linestyle=':',alpha=0.5)
ax1.plot([0,-1],[0,0],[0,0],color='k',linestyle=':',alpha=0.5)
ax1.plot([0,0],[0,-1],[0,0],color='k',linestyle=':',alpha=0.5)
ax1.plot([0,0],[0,0],[0,-1],color='k',linestyle=':',alpha=0.5)
ax1.set_title('|0>-[X]--[B]=', fontsize=10)
ax2.plot_wireframe(xaxis,yaxis,zaxis,rstride=10,cstride=60,color='grey',alpha=0.5)
ax2.plot(np.sin(qp),np.cos(qp),0,color='k',linestyle='--',alpha=0.5)
ax2.plot([0,1],[0,0],[0,0],color='k',linestyle=':',alpha=0.5)
ax2.plot([0,0],[0,1],[0,0],color='k',linestyle=':',alpha=0.5)
ax2.plot([0,0],[0,0],[0,1],color='k',linestyle=':',alpha=0.5)
ax2.plot([0,-1],[0,0],[0,0],color='k',linestyle=':',alpha=0.5)
ax2.plot([0,0],[0,-1],[0,0],color='k',linestyle=':',alpha=0.5)
ax2.plot([0,0],[0,0],[0,-1],color='k',linestyle=':',alpha=0.5)
ax2.set_title('|0>-[H]-[S]--[B]=', fontsize=10)
ax3.plot_wireframe(xaxis,yaxis,zaxis,rstride=10,cstride=60,color='grey',alpha=0.5)
ax3.plot(np.sin(qp),np.cos(qp),0,color='k',linestyle='--',alpha=0.5)
ax3.plot([0,1],[0,0],[0,0],color='k',linestyle=':',alpha=0.5)
ax3.plot([0,0],[0,1],[0,0],color='k',linestyle=':',alpha=0.5)
ax3.plot([0,0],[0,0],[0,1],color='k',linestyle=':',alpha=0.5)
ax3.plot([0,-1],[0,0],[0,0],color='k',linestyle=':',alpha=0.5)
ax3.plot([0,0],[0,-1],[0,0],color='k',linestyle=':',alpha=0.5)
ax3.plot([0,0],[0,0],[0,-1],color='k',linestyle=':',alpha=0.5)
ax3.set_title('|0>-[H]-[Sd]--[B]=', fontsize=10)
ax4.plot_wireframe(xaxis,yaxis,zaxis,rstride=10,cstride=60,color='grey',alpha=0.5)
ax4.plot(np.sin(qp),np.cos(qp),0,color='k',linestyle='--',alpha=0.5)
ax4.plot([0,1],[0,0],[0,0],color='k',linestyle=':',alpha=0.5)
ax4.plot([0,0],[0,1],[0,0],color='k',linestyle=':',alpha=0.5)
ax4.plot([0,0],[0,0],[0,1],color='k',linestyle=':',alpha=0.5)
ax4.plot([0,-1],[0,0],[0,0],color='k',linestyle=':',alpha=0.5)
ax4.plot([0,0],[0,-1],[0,0],color='k',linestyle=':',alpha=0.5)
ax4.plot([0,0],[0,0],[0,-1],color='k',linestyle=':',alpha=0.5)
ax4.set_title('|0>-[H]-[Z]--[B]=', fontsize=10)

# label x, y, |0>, |1>
ax0.text(0.1,0,1.3,r"$|0\rangle$")
ax0.text(0.1,0,-1.5,r"$|1\rangle$")
ax0.text(1.3,0,0,r"$x$")
ax0.text(0,1.2,0,r"$y$")
ax1.text(0.1,0,1.3,r"$|0\rangle$")
ax1.text(0.1,0,-1.5,r"$|1\rangle$")
ax1.text(1.3,0,0,r"$x$")
ax1.text(0,1.2,0,r"$y$")
ax2.text(0.1,0,1.3,r"$|0\rangle$")
ax2.text(0.1,0,-1.5,r"$|1\rangle$")
ax2.text(1.3,0,0,r"$x$")
ax2.text(0,1.2,0,r"$y$")
ax3.text(0.1,0,1.3,r"$|0\rangle$")
ax3.text(0.1,0,-1.5,r"$|1\rangle$")
ax3.text(1.3,0,0,r"$x$")
ax3.text(0,1.2,0,r"$y$")
ax4.text(0.1,0,1.3,r"$|0\rangle$")
ax4.text(0.1,0,-1.5,r"$|1\rangle$")
ax4.text(1.3,0,0,r"$x$")
ax4.text(0,1.2,0,r"$y$")

# camera angle
ax0.elev = 40
ax0.azim = 45
ax1.elev = 40
ax1.azim = 45
ax2.elev = 40
ax2.azim = 45
ax3.elev = 40
ax3.azim = 45
ax4.elev = 40
ax4.azim = 45

ax0.scatter(blochxyz0[0],blochxyz0[1],blochxyz0[2], s=80, c='red', alpha=0.5)
ax0.plot([0,blochxyz0[0]],[0,blochxyz0[1]],[0,blochxyz0[2]], linewidth=2.0, color='red', alpha=0.5)
ax1.scatter(blochxyz1[0],blochxyz1[1],blochxyz1[2], s=80, c='orange', alpha=0.5)
ax1.plot([0,blochxyz1[0]],[0,blochxyz1[1]],[0,blochxyz1[2]], linewidth=2.0, color='orange', alpha=0.5)
ax2.scatter(blochxyz2[0],blochxyz2[1],blochxyz2[2], s=80, c='green', alpha=0.5)
ax2.plot([0,blochxyz2[0]],[0,blochxyz2[1]],[0,blochxyz2[2]], linewidth=2.0, color='green', alpha=0.5)
ax3.scatter(blochxyz3[0],blochxyz3[1],blochxyz3[2], s=80, c='blue', alpha=0.5)
ax3.plot([0,blochxyz3[0]],[0,blochxyz3[1]],[0,blochxyz3[2]], linewidth=2.0, color='blue', alpha=0.5)
ax4.scatter(blochxyz4[0],blochxyz4[1],blochxyz4[2], s=80, c='purple', alpha=0.5)
ax4.plot([0,blochxyz4[0]],[0,blochxyz4[1]],[0,blochxyz4[2]], linewidth=2.0, color='purple', alpha=0.5)

ax0.scatter([0],[0],[0], c='black', s=10)
ax1.scatter([0],[0],[0], c='black', s=10)
ax2.scatter([0],[0],[0], c='black', s=10)
ax3.scatter([0],[0],[0], c='black', s=10)
ax4.scatter([0],[0],[0], c='black', s=10)

plt.tight_layout()
plt.show()
