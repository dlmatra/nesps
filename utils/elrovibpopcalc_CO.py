def elrovibpopcalc_COfunc(tgas=None, ncollp=None, jnufilestar='', nlinesjnufilestar=None, jnufilerot=None, datafile='',  linewithntrcoll=None, distpc=None, distau=None, massmearth=None, levupint=None, si=0, fluxJykmsinp=None):

	### DESCRIPTION
	#The function calculates the fractional population of supplied energy levels by solving the statistical equilibrium system of linear equations (Eq. 11 of Matra et al. 2015, MNRAS, 447, 3936), for a given gas kinetic temperature, collider density (currently electrons) and mean intensity of the radiation field. It needs a list of energy levels and transitions in LAMDA format, and can output all fractional level populations, or fractional population for a specified transition, or line fluxes for a given mass, or masses for a given observed line flux.

	###(Current) ASSUMPTIONS
	#Collisions with electrons, between neighbouring ROTATIONAL levels only. Rates from Dickinson and Richards 1975.
	#Dust disk mean intensity assumed to be significant as a pumping mechanism for ROTATIONAL levels only. No dust disk pumping to vibrational or electronic levels above ground.
	#UV ISRF automatically included in code (leads to fluorescence); assuming Draine 1978+extension field from https://home.strw.leidenuniv.nl/~ewine/photo/radiation_fields.html
	#CMB automatically included in radiation field, on top of stellar, dust and ISRF component.
	#Stimulated emission included in statistical equilibrium, but not necessarily in the line flux calculation. Toggle on or off by setting stimem parameter below as 1 or 0.

	### INPUTS
	#distau: distance of gas from the star in AU (for fluorescence due to stellar emission calculation, to rescale flux at Earth to flux at gas species of interest)
	#distpc: distance of star from Earth in pc (for fluorescence due to stellar emission calculation, to rescale flux at Earth to flux at gas species of interest)
	#ncollp: density of collisional partners (electrons only for now)
	#jnufilestar: numpy save file containing stellar flux as observed in Jy from Earth which needs continuous wavelength (micron) coverage between 0.088 and 5.5 micron to cover all rovibronic and rovibrational transitions, and that can be called with: wavuvum, fluxuvJy=np.load(jnufilestar)
	#nlinesjnufilestar: number of lines in flux file; not needed if file given is numpy array.
	#jnufilerot: file containing dust disk mean intensity in Jy/sr at all 261 rotational transition frequencies, listed in file 'freq_rottransCO_GHz.txt' for ease of calculating dust mean intensities outside of this code. Needs to have size 261 or routine breaks
	#tgas: kinetic temperature of the gas
	#levupint: upper level of the transition of interest, only needed if massmearth or fluxJykmsinp are specified
	#massmearth: mass of gas in Earth masses - only if you want a flux calculation
	#fluxJykmsinp: integrated line flux of gas in Jy km/s - only if you want the mass calculated. NB: the si flag in the input commands only works when flux is the output and not the input. So always give Jy km/s as input for now.
	#datafile: file containing radiative and collisional transitions with collisional partner of choice in LAMDA format
	#linewithntrcoll: line in LAMDA molecular data file containing the number of collisional transitions considered. Not necessary at the moment as we are calculating electron-CO coefficients on the fly.

	### OUTPUTS
	# If interested in all levels, set levupint='all'. If interested in a specific level, set levupint equal to the upper energy level of interest. Need to exactly match the upper level ID from the LAMDA-style energy levels file provided. KNOWN ISSUE: what happens in flux or mass calculation when upper level is non-unique?
	# - if massmearth provided, si=0, returning line flux in Jy km/s for given transition or all transitions, assuming optically thin emission.
	# - if massmearth provided, si=1, returning line flux in W m**-2 for given transition or all transitions, assuming optically thin emission.
	# - if fluxjykms provided, (si irrelevant) returning optically thin mass in Earth masses.
	# - if neither massmearth or fluxjykms provided, returning fractional population of given level or all levels (unitless).

	### Import Packages
	import matplotlib.pyplot as pl
	import numpy as np
	import sys
	import scipy
	from scipy.interpolate import interp1d
	import calc_electcollcoeff
	import importlib
	importlib.reload(calc_electcollcoeff)
	
	### Import Constants
	from scipy.constants import h,c,k
	

	## tgas and ncollp are always needed so put a control straight away in case they are not provided
	if (not tgas) or (not ncollp) or (not levupint):
		print('Error!' )
		print('Gas kinetic temperature (tgas) or collisional partner density (ncollp) or upper level of interest (levup) were not set!')
		sys.exit()
			

	### Read in line transition data 
	fname=datafile
	try:
		rfile = open(fname, 'r')
	except:
		print('Error!') 
		print('Excitation file was not found!')
		sys.exit() 
	allfilelines=rfile.readlines() 	
	rfile.close()
	rfile = open(fname, 'r')
	
	
	### Read energy level file in LAMDA-style (dum are empty lines)
	dum=rfile.readline()
	#Name of species
	species=rfile.readline().split()[0]
	dum=rfile.readline()
	#Molecular weight of species
	mwt=float(rfile.readline().split()[0])
	dum=rfile.readline()
	#Total number of energy levels included
	n_levels=int(rfile.readline().split()[0])
	dum=rfile.readline()
	
	#Energy level ID
	levid=np.zeros(n_levels, dtype=int)
	#Level energy in cm**-1
	energycmmin1=np.zeros(n_levels, dtype=float)
	#Level degeneracy
	wt=np.zeros(n_levels, dtype=float)
	#Electronic quantum number ID
	qnel=np.zeros(n_levels, dtype=int)
	#Vibrational quantum number ID
	qnvib=np.zeros(n_levels, dtype=int)
	#Rotational quantum number ID
	qnrot=np.zeros(n_levels, dtype=int)
	
	#Read in all energy level information
	for i in np.arange(n_levels):
		levid[i], energycmmin1[i], wt[i], qnel[i], qnvib[i], qnrot[i] = rfile.readline().split()
		#levid[i], energycmmin1[i], wt[i], qnrot[i] = rfile.readline().split()    #If using rotational transitions only
	
	
	### Read data on all transitions listed in LAMDA file
	dum=rfile.readline()
	#Total number of transitions
	ntr=int(rfile.readline().split()[0])	
	dum=rfile.readline()
	#Transition ID
	trid=np.zeros(ntr, dtype=int)
	#Upper energy level ID
	levup=np.zeros(ntr, dtype=int)
	#Lower energy level ID
	levdwn=np.zeros(ntr, dtype=int)
	#Einstein A coefficient of transition (s**-1)
	einstA=np.zeros(ntr, dtype=float)
	#Frequency of transition in GHz
	freqghz=np.zeros(ntr, dtype=float)
	#Energy of upper level of transition in Kelvin (E/k)
	eupink=np.zeros(ntr, dtype=float)
	
	#Read in all transition information
	for i in np.arange(ntr):
		trid[i], levup[i], levdwn[i], einstA[i], freqghz[i], eupink[i] = rfile.readline().split()
	rfile.close()
	
	### Read collisional transition data from LAMDA file, if necessary.
	if not linewithntrcoll:
		print('Error!' )
		print('Line containing number of collisional transitions for collider of interest was not found!')
		sys.exit() 
	ntrcollelect=int(allfilelines[linewithntrcoll-1])
	neleccolltemps=int(allfilelines[linewithntrcoll+1])
	eleccolltemps=np.zeros(neleccolltemps, dtype=float)
	for i in np.arange(neleccolltemps):
		eleccolltemps[i]=allfilelines[linewithntrcoll+3].split()[i]
	
	trcollelectid=np.zeros(ntrcollelect, dtype=int)
	colllevup=np.zeros(ntrcollelect, dtype=int)
	colllevdwn=np.zeros(ntrcollelect, dtype=int)
	collrates=np.zeros((ntrcollelect,neleccolltemps), dtype=float)
	
	# Commented out below here because tabulated electron-CO collision rates are wrong
	#for t in np.arange(neleccolltemps):
	#	for i in np.arange(ntrcollelect):
	#		if t==0:
	#			trcollelectid[i]=allfilelines[linewithntrcoll+5+i].split()[0]
	#			colllevup[i]=allfilelines[linewithntrcoll+5+i].split()[1]
	#			colllevdwn[i]=allfilelines[linewithntrcoll+5+i].split()[2]
	#		collrates[i,t]=allfilelines[linewithntrcoll+5+i].split()[3+t]
	#tgas=float(tgas)
	#collratesattgas=np.zeros(ntrcollelect, dtype=float)
	#for i in np.arange(ntrcollelect):
	#	interpcollrates=interp1d(eleccolltemps,collrates[i,:],kind='linear')
	#	collratesattgas[i]=interpcollrates(tgas)
	#ncoll=ncollp #in cm^-3
	#np.save(collratesav, 'collratesav.npy')
	#collratesav=collrates[np.intersect1d(np.where(colllevup==11)[0],np.where(colllevdwn==10)[0]),:]
	#np.save('collratesav.npy', collratesav)
	#np.save('tempscollsav.npy', eleccolltemps)
	
	
	# Workaround here to take external electron-CO rate coefficients rather than tabulated (wrong) ones
	#for i in np.arange(ntrcollelect):
	#	for t in np.arange(neleccolltemps):			
	#		if t==0:
	#			trcollelectid[i]=allfilelines[linewithntrcoll+5+i].split()[0]
	#			colllevup[i]=allfilelines[linewithntrcoll+5+i].split()[1]
	#			colllevdwn[i]=allfilelines[linewithntrcoll+5+i].split()[2]
	#		if colllevup[i]-colllevdwn[i]==1:
	#			collrates[i,t]=calc_electcollcoeff.calc_electcollcoeff(eleccolltemps[t],colllevup[i]-1,colllevdwn[i]-1)			
	#tgas=float(tgas)
	#collratesattgas=np.zeros(ntrcollelect, dtype=float)
	#for i in np.arange(ntrcollelect):
	#	interpcollrates=interp1d(eleccolltemps,collrates[i,:],kind='linear')
	#	collratesattgas[i]=interpcollrates(tgas)
	#ncoll=ncollp #in cm^-3
	#np.save(collratesav, 'collratesav.npy')
	#collratesav=collrates[np.intersect1d(np.where(colllevup==11)[0],np.where(colllevdwn==10)[0]),:]
	#np.save('collratesav.npy', collratesav)
	#np.save('tempscollsav.npy', eleccolltemps)

	#Calculate collisional rate coefficients for electron-CO collisions on the fly, for a given gas temperature. Including only rotational levels from LAMDA file (ID<31 in LAMDA file), and assuming only collisional transitions between neighbouring rotational levels are significant (Dickinson and Richards 1975). Remember that J of a given rotational level in LAMDA file is the level ID minus 1!
	n=0
	tgas=float(tgas)
	collratesattgas=np.zeros(ntrcollelect, dtype=float)
	for i in np.arange(ntrcollelect):
		trcollelectid[i]=allfilelines[linewithntrcoll+5+i].split()[0]
		colllevup[i]=allfilelines[linewithntrcoll+5+i].split()[1]
		colllevdwn[i]=allfilelines[linewithntrcoll+5+i].split()[2]
		if (colllevup[i]-colllevdwn[i])==1 and (colllevup[i]<31):
			n+=1
			collratesattgas[i]=calc_electcollcoeff.calc_electcollcoeff(tgas,colllevup[i]-1,colllevdwn[i]-1)	
	ncoll=ncollp


	### Read in stellar UV flux from ASCII table, if necessary
	##flname='bPicNormFlux1AU.dat'
	#if jnufilestar:
	#	flname=jnufilestar
	#	#nlines=261549
	#	if not nlinesjnufilestar:
	#		print 'Error!' 
	#		print 'Stellar flux file name was set but the number of lines in this file was not set!'
	#		sys.exit()
	#	nlines=int(nlinesjnufilestar)
	#	try:
	#		rfile = open(flname, 'r')
	#	except:
	#		print 'Error!' 
	#		print 'Stellar flux file name was set but the file was not found!'
	#		sys.exit()
	#	if (not distpc) or (not distau):
	#		print 'Error!' 
	#		print 'Stellar flux file name (containing observed fluxes in Jy) was set but either the distpc or distau were not set!'
	#		sys.exit()
	#	wavuvum=np.zeros(nlines)
	#	fluxuvJy=np.zeros(nlines)
	#	for i in np.arange(nlines):
	#		wavuvum[i], fluxuvJy[i]=rfile.readline().split()
	#	
	#	#wavuvum=wavuv*1e-4
	#	freqjnughz=c/(wavuvum*1e-6)/1e9
	#	fluxuvSI1AU=fluxuvJy*1e-26*((distpc*2.0626*1e5)**2.0)
	#	#(10**logfluxuv)/2.99792458e21*(wavuv**2.0)
	#	fluxuvSIat85au=fluxuvSI1AU/(distau**2.0)
	#	
	#	freqjnughz=freqjnughz[::-1]
	#	fluxuvSIat85au=fluxuvSIat85au[::-1]
	#	
	#	### Check that flux makes sense as seen from Earth in Jy
	#	#pl.figure()
	#	#pl.plot(wavuvum, fluxuvSI1AU*1e26/((distpc*2.0626*1e5)**2.0))
	#	#pl.yscale('log')
	#	#pl.xscale('log')
	#

	### Read in stellar flux at all wavelengths, if necessary, from numpy array
	if jnufilestar:
		wavuvum, fluxuvJy=np.load(jnufilestar)
		freqjnughz=c/(wavuvum*1e-6)/1e9
		fluxuvSI1AU=fluxuvJy*1e-26*((distpc*2.0626*1e5)**2.0)#/1000.0
		fluxuvSIat85au=fluxuvSI1AU/(distau**2.0)
		freqjnughz=freqjnughz[::-1]
		fluxuvSIat85au=fluxuvSIat85au[::-1]

	### Read in disk mean intensity for rotational levels, if necessary, from numpy array. Currently, if included, this needs to be pre-computed at the frequencies freqghz[:261] (see documentation at the top). That is, only pumping of rotational levels by dust emission is included here.
	if jnufilerot:
		freqjnughzrotonly=freqghz[:261]
		fluxrotonlyJysr=np.load(jnufilerot)
		if (fluxrotonlyJysr.size != freqjnughzrotonly.size):
			print('Error!')
			print('Number of input mean intensities for rotational transitions (far-IR to mm) does not match the number of frequencies for these transitions provided!')
			sys.exit()

	### Here read in UV intensity from the ISRF
	#Read interstellar Draine 1978 +extension field flux file
	rfile = open('./ISRF.dat', 'r')
	dum=rfile.readline()
	dum=rfile.readline()
	dum=rfile.readline()
	dum=rfile.readline()
	nlinesisrf=1909
	wavisrfnm=np.zeros(nlinesisrf)
	fluxisrfweird=np.zeros(nlinesisrf)
	#Read wavelengths in nm and fluxes in photons cm^-2 s^-1 nm^-1
	for i in np.arange(nlinesisrf):
		wavisrfnm[i], fluxisrfweird[i] = rfile.readline().split()
	wavisrfang=wavisrfnm*10.0
	fluxisrf=fluxisrfweird/10.0 # Here have converted to photons cm^-2 s^-1 A^-1
	fluxisrfjy=fluxisrf*6.63e-4*wavisrfang #Here converted to Jy
	fluxisrfSI=fluxisrfjy*1e-26 #Here converted to SI units, W m**-2 Hz**-1
	freqGHzISRF=c/(wavisrfnm*1e-9)/1e9 #Here converted wavelength in nm to frequency in GHz.
					

	
	#Set up all matrices in preparation of solving statistical equilibrium. The statistical equilibrium (system of linear equations) is viewed as a matrix multiplication between an NxN matrix (called S here) containing all the rates, and a column vector containing all the fractional populations x_0, x_1, ... x_N. The right hand side (called rhs here) of the matrix equation is a column vector with 0 everywhere. The normalization condition is then imposed in the first equation so that x_0+x_1+...+x_N=1, in other words all elements of the first row of S are set to 1, and the first element of the rhs column vector is set to 1.
	#S is an NxN matrix where N is the number of energy levels for the species in question. Every row of the matrix represents one of the linear equations of statistical equilibrium for a given level i. Every element of that row, say j, then is the transition rate into (+ sign) or out of (- sign) level i from level j. As only few levels j contribute to the population of level i, most elements of any given row are 0 - this is a sparse matrix, and mostly populated near the diagonal.
	#Most of the work is in building the matrix S from equal sized matrices containing each type of transition, A for spontaneous emission rates, BJstim for stimulated emission, BJabs for radiative absorption, K for collisional rates. S is then simply the sum of all these.

	#Create empty matrices
	A=np.zeros((n_levels, n_levels))
	BJstim=np.zeros((n_levels, n_levels))
	BJabs=np.zeros((n_levels, n_levels))
	K=np.zeros((n_levels, n_levels))
	Jstim=np.zeros((n_levels, n_levels))
	Jabs=np.zeros((n_levels, n_levels))
	
	#Calculate CMB radiation field at frequencies corresponding to all transitions, and initialise mean intensity jnu of radiation field to CMB alone
	T_cmb		=	float(2.72548)			#temperature of CMB
	jnu=2.*h*((freqghz*1e9)**3.0)/((c**2.0)*(np.exp(h*freqghz*1e9/(k*T_cmb))-1.))	#Bnu(CMB)

	#Interpolate ISRF UV field at frequencies corresponding to all transitions, and add it to radiation field jnu
	fintISRF=interp1d(freqGHzISRF[::-1], fluxisrfSI[::-1], kind='linear')
	jnu[(freqghz>freqGHzISRF.min()) & (freqghz<freqGHzISRF.max())]+=1.0/4.0/np.pi*fintISRF(freqghz[(freqghz>freqGHzISRF.min()) & (freqghz<freqGHzISRF.max())]) #Note division by 4pi here as we want mean intensity rather than flux

	#If we are including the stellar mean intensity, also interpolate the stellar flux to the frequencies corresponding to all transitions, and convert to mean intensity
	if jnufilestar:
		fint=interp1d(freqjnughz, fluxuvSIat85au, kind='linear')
		#print('Make sure radiation field covers well the UV out to '+str(c/freqghz.max()/1e9*1e6)+' micron - right now it goes out to '+str(c/freqjnughz.max()/1e9*1e6)+' micron')
		jnu[np.intersect1d(np.where(freqghz>freqjnughz.min())[0],np.where(freqghz<freqjnughz.max())[0])]+=1.0/4.0/np.pi*fint(freqghz[np.intersect1d(np.where(freqghz>freqjnughz.min())[0],np.where(freqghz<freqjnughz.max())[0])]) #Note division by 4pi here as we want mean intensity rather than flux

	#If we are including the dust disk mean intensity, also interpolate that to the frequencies corresponding to all transitions. This is assumed to have mean intensity units already
	if jnufilerot:
		jnu[:261]+=fluxrotonlyJysr*1e-26	

	#Plot final mean intensity if needed
	#pl.figure()
	#pl.plot(c/freqghz/1e9*1e6, jnu, '+')
	#pl.xscale('log')
	#pl.yscale('log')
	#print np.max(colllevup)

	### Prepare matrices for radiative transitions. Go through all the allowed radiative transitions from LAMDA file, and 
	### - Take Einstein As of each transitions and populate the spontaneous emission matrix A
	### - Use Einstein As to calculate downward Einstein Bs and multiply by the mean intensity to populate the stimulated emission matrix BJstim
	### - Use Einstein As to calculate upward Einstein Bs and multiply by the mean intensity to populate the radiative absorption matrix BJabs
	for i in np.arange(ntr):
		if (einstA[i] !=0.0):
			A[np.where(levid==levdwn[i])[0], np.where(levid==levup[i])[0]]=einstA[i]	#Spontaneous emission into the level given by the row
			BJstim[np.where(levid==levdwn[i])[0], np.where(levid==levup[i])[0]]=einstA[i]*(c**2.0)/(2.0*h*((freqghz[i]*1e9)**3.0))*jnu[i]	#Stimulated emission into the level given by the row
			BJabs[np.where(levid==levup[i])[0], np.where(levid==levdwn[i])[0]]=wt[np.where(levid==levup[i])[0]]/wt[np.where(levid==levdwn[i])[0]]*einstA[i]*(c**2.0)/(2.0*h*((freqghz[i]*1e9)**3.0))*jnu[i]	#Absorption into the level given by the row (different from one above)
	
	### Prepare matrices for collisional transitions. Go through all the collisional transitions from the LAMDA file, and
	### - Take collisional rate obtained/calculated in code above, multiply by the density of collisional partners ncoll, and place as the downward collisional transition rate into as a positive rate populating the lower level of the transition.
	### - Take the same downward rate, and use the detailed balance equation for collisional rate coefficients to calculate the upward rate. Place upward rate as a positive rate populating the upper level of the transition.
	for i in np.arange(ntrcollelect):
		if (collratesattgas[i] != 0.0):
			K[np.where(levid==colllevdwn[i])[0], np.where(levid==colllevup[i])[0]]=collratesattgas[i]*ncoll		#Collisional downward transition into the level given by the row (positive rate, as it will populate that level)
			K[np.where(levid==colllevup[i])[0], np.where(levid==colllevdwn[i])[0]]=collratesattgas[i]*ncoll*wt[np.where(levid==colllevup[i])[0]]/wt[np.where(levid==colllevdwn[i])[0]]*np.exp(-float(1.43877736)*np.abs(energycmmin1[np.where(levid==colllevup[i])[0]]-energycmmin1[np.where(levid==colllevdwn[i])[0]])/tgas)		#Collisional upward transition into the level given by the row (positive rate, as it will populate that level)

	#Check matrix, as an image to make sure it makes sense	
	#pl.figure()
	#pl.imshow(np.log10(np.abs(K)), origin='lower')	

	#Above in A, BJstim, BJabs, and K we only accounted for rates that populate each level given by a row in the matrix. Now we account for all the rates that depopulate each level given by a row in the matrix. These naturally all go in the diagonal of each matrix, as the rate of 'depopulation' needs to be multiplied by the fractional population of the level in question.
	for i in np.arange(n_levels):
		if i!=0: A[i,i]=-np.sum(A[:,i]) #Sum all the spontaneous emission rates along the column corresponding to the current row. The negative of that will be the depopulation rate through spontaneous emission.
		if i!=0: BJstim[i,i]=-np.sum(BJstim[:,i]) #Similar to the above, but for stimulated emission
		if (i!=(n_levels-1)): BJabs[i,i]=-np.sum(BJabs[:,i]) #Similar to the above, but for radiative absorption
		K[i,i]=-np.sum(K[:,i]) #Similar to the above, but for collisional rates.
	
	#Create master matrix with all coefficients
	S=A+BJstim+BJabs+K
	#Substitute first row to be the normalising equation, where the coefficients are all 1
	S[0,:]=1.0
	#Prepare right hand side of matrix equation as column vector with 1 in the first row, and zeros everywhere else
	rhs=np.zeros((n_levels))
	rhs[0]=1.0
	#Solve system of linear equation with scipy.linalg.lu_solve.
	fracpoplev=scipy.linalg.lu_solve(scipy.linalg.lu_factor(S), np.transpose(rhs))
	#Have solved the statistical equilibrium. fracpoplev is an array with the fractional populations of all N levels.
	
	### Calculate flux from input mass (not needed, can just return populations)
	stimem=1 #Toggle stimulated emission on or off
	if levupint=='all':
		fluxwm2=np.zeros(ntr)
		fluxjykms=np.zeros(ntr)
		fracpoplevup=np.zeros(ntr)
		mass=np.zeros(ntr)
		if massmearth:
			for i in np.arange(ntr):
				fracpoplevup[i]=fracpoplev[np.where(levid==levup[i])[0]]
				if stimem: fluxwm2[i]=massmearth*5.9736e24*h*freqghz[i]*1e9*einstA[i]*(1.0+(c**2.0)/(2.0*h*((freqghz[i]*1e9)**3.0))*jnu[i])*fracpoplevup[i]/(4.0*np.pi*1.67e-27*mwt*((distpc*3.0857e16)**2.0))
				else: fluxwm2[i]=massmearth*5.9736e24*h*freqghz[i]*1e9*einstA[i]*(1.0)*fracpoplevup[i]/(4.0*np.pi*1.67e-27*mwt*((distpc*3.0857e16)**2.0))
			if si:
				return fluxwm2
			else:
				fluxjykms=fluxwm2/freqghz/1e9*c/1e3*1e26
				return fluxjykms
		else: 
			if fluxJykmsinp:
				for i in np.arange(ntr):
					fracpoplevup[i]=fracpoplev[np.where(levid==levup[i])[0]]
					if stimem: mass[i]=fluxJykmsinp*freqghz*1e9/c*1e3/1e26/(5.9736e24*h*freqghz[i]*1e9*einstA[i]*(1.0+(c**2.0)/(2.0*h*((freqghz[i]*1e9)**3.0))*jnu[i])*fracpoplevup[i]/(4.0*np.pi*1.67e-27*mwt*((distpc*3.0857e16)**2.0)))
					else: mass[i]=fluxJykmsinp*freqghz*1e9/c*1e3/1e26/(5.9736e24*h*freqghz[i]*1e9*einstA[i]*(1.0)*fracpoplevup[i]/(4.0*np.pi*1.67e-27*mwt*((distpc*3.0857e16)**2.0)))
				return mass
			else:
				return fracpoplev				
	else:
		if massmearth:
			if stimem: fluxwm2=massmearth*5.9736e24*h*freqghz[np.where(levup==(levupint+1))[0]]*1e9*einstA[np.where(levup==(levupint+1))[0]]*(1.0+(c**2.0)/(2.0*h*((freqghz[np.where(levup==(levupint+1))[0]]*1e9)**3.0))*jnu[np.where(levup==(levupint+1))[0]])*fracpoplev[levupint]/(4.0*np.pi*1.67e-27*mwt*((distpc*3.0857e16)**2.0))
			else: fluxwm2=massmearth*5.9736e24*h*freqghz[np.where(levup==(levupint+1))[0]]*1e9*einstA[np.where(levup==(levupint+1))[0]]*(1.0)*fracpoplev[levupint]/(4.0*np.pi*1.67e-27*mwt*((distpc*3.0857e16)**2.0))
			if si:
				return fluxwm2
			else:
				fluxjykms=fluxwm2/freqghz[np.where(levup==(levupint+1))[0]]/1e9*c/1e3*1e26
				return fluxjykms
		else:
			if fluxJykmsinp:
				if stimem: mass=fluxJykmsinp*freqghz[np.where(levup==(levupint+1))[0]]*1e9/c*1e3/1e26/(5.9736e24*h*freqghz[np.where(levup==(levupint+1))[0]]*1e9*einstA[np.where(levup==(levupint+1))[0]]*(1.0+(c**2.0)/(2.0*h*((freqghz[np.where(levup==(levupint+1))[0]]*1e9)**3.0))*jnu[np.where(levup==(levupint+1))[0]])*fracpoplev[levupint]/(4.0*np.pi*1.67e-27*mwt*((distpc*3.0857e16)**2.0)))
				else: mass=fluxJykmsinp*freqghz[np.where(levup==(levupint+1))[0]]*1e9/c*1e3/1e26/(5.9736e24*h*freqghz[np.where(levup==(levupint+1))[0]]*1e9*einstA[np.where(levup==(levupint+1))[0]]*(1.0)*fracpoplev[levupint]/(4.0*np.pi*1.67e-27*mwt*((distpc*3.0857e16)**2.0)))		
				return mass
			else:
				return fracpoplev[levupint]
