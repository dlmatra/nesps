#*************************************************************************************************************************************
#Calculate collisional rate coefficients of CO with ELECTRONS, using formula from Dickinson & Richards 1975, valid to about 30% at T<5000K
#Collisional excitation by electrons only significant for neighbouring levels. 
#Eq. 2.23 of Dickinson 1975
#************************************************************************************************************************************

def calc_electcollcoeff(T, jlow, jup):

	import numpy as np
	
	if (np.abs(jlow-jup) !=1):
		print('Coefficients are only calculated for neighbouring levels!')
		return 0.0
	
	T=float(T)
	jl=float(np.min([jlow,jup]))
	ju=float(np.max([jlow,jup]))
	B0=5.7635968e10*3.34e-11 		#rotational constant, in cm^-1
	dipmom=0.11011				#molecular dipole moment, in Debyes
	deltaE=2.48e-4*B0*(ju)			#threshold energy, in eV
	beta=11600./T				#beta=1/kbT, in 1/eV
	A=2.470*(dipmom**2.0)*ju/(2.0*jl+1.0)		#constant for calculation
	C=9.08e3/(B0*ju)			#constant for calculation
	# stuff below added Aug 16
	B0J=B0*1.98630e-23					#rotational constant, in J
	Eu=ju*(ju+1)*B0J/1.38e-23			#energy of upper level in K
	El=jl*(jl+1)*B0J/1.38e-23			#energy of upper level in K
	
	alpha=(1.44e-6/np.sqrt(T))*A*np.exp(-beta*deltaE)*np.log(C*deltaE+(C/beta)*np.exp(-0.577/(1.0+2.0*beta*deltaE)))		#coll rate coefficient in cm^3 s^-1
	alphadown=alpha*(2*jl+1)/(2*ju+1)*np.exp((Eu-El)/T)
	
	if (jlow < jup): return alpha
	if (jlow > jup): return alphadown
