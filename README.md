# nesps
Non-LTE Excitation Solver for Planetary Systems

Currently solving the statistical equilibrium equations to do non-LTE excitation of the CO molecule, including fluorescence. 

Awaiting more organised documentation, please see the simple NLTE exercise tutorial for classic usage, and refer to the detailed documentation within the main code elrovibpopcalc_CO.py under /utils.

References:
*NLTE code*: Matrà et al. 2015, MNRAS, 447, 3936; Matrà et al. 2018, ApJ, 853, 147 [for inclusion of fluorescence calculation]

*CO line list and collision rates*: Thi et al. 2012, A&A, 551, 49; big thanks to Gianni Cataldi for correcting an issue with this list.

*electron-CO collision rates [currently implemented]*: Dickinson and Richards 1975, J. Phys. B: Atom. Mol. Phys., 8, 2846

*Interstellar radiation field*: Draine 1978, ApJS, 36, 595 [with extension of van Dishoeck and Black 1982, ApJ, 258, 533 at longer wavelengths than 200nm, see https://home.strw.leidenuniv.nl/~ewine/photo/radiation_fields.html]

*beta Pic stellar spectrum used in example*: 
Brandeker, A. private communication, compiled from observed spectra and a model spectrum, see Matrà et al. 2018 above for references.
