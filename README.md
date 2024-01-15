# JP-metric-Parameter-Estimation

In the post-merger phase of a black hole binary system, the remnant object is a perturbed black hole emitting gravitational radiation in the form of Gravitational Waves (quasi-normal modes) before admitting a stable state. Observations of gravitational waves can thus be used to test and constrain deviations from Einstein’s theory of gravity. The no-hair theorem says that only 2 parameters are required to characterize an astrophysical black hole described by the Kerr metric; mass and spin. However, in the post-Kerr approximation, the metric incorporates non-Kerr parameters. We assume that the observed ringdown can be described by the Johanssen-Psaltis (JP) metric and use LVK data to constrain the deviation parameter.

This code calculates the values that the JP parameter epsilon_3 takes, given GW190521 LIGO data as an example, with the ultimate goal of trying to put constraints on epsilon_3.  

Repository fodlers are organized as follows:

1- Agnostic Data: Ringdown frequency and damping time agnostic posterior distributions obtained from LVK strain data of the two events GW150914 and GW190521. Configuration files for PyCBC inference are also provided, to be adjusted according the local files paths of the user. 
2- Codes: Parameter estimation in the JP geometry, defined python functions compute a grid of (f,tau) from the parameter space (mass, spin, epsilon_3). A multiprocessing code chunk is use to produce posterior distributions using the agnostic (f,tau) data distributions. The code with file name 'Parameter Estimation with Multiprocessing.py' can be adjusted for the specifc event data files in 'Agnostic Data' and prior ranges must be adjusted accordingly.
3- Posterior Data: Posterior samples of (mass, spin, epsilon_3) obtained by running the code from 'Codes' for the two events and stored in their respective HDF files. 
4- Figures: Relevant figures produced using JP posterior samples.

Details about the subject matter can be found in arXiv: 2401.06049.
