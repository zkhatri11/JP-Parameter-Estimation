import math,scipy,numpy,inspect,lal,time,h5py,random,scipy.stats,corner
from sympy import *
from sympy import init_printing
from sympy import solve,symbols
import matplotlib.pyplot as plt
import multiprocessing as mp
from pycbc.inference import io
from tqdm import tqdm
from multiprocessing import Pool, Manager
from multiprocessing.pool import ThreadPool
from scipy.stats import norm

def common(y,spin,epsilon): 
    ''' This is a common term that shows up in most
        of the algebraic expressions, frequently under a square root.
        
        This term turns out to be negative for some values of spin 
        and epsilon which causes RuntimeError because the square root
        becomes complex. We do not need complex numbers and therefore we discard 
        the values of spin and epsilon using the if condition; C>=0. 
        If the condition is False, the function returns None and this output 
        is used as a condition in the frequency and damping time 
        expressions to discard complex numbers.
    '''
    
    C = (9*epsilon**2*spin**2)/y**8 - 6*epsilon/y**3 + 16*epsilon/y**4 + 4/y

    if C>=0:
        return C
    return None

def func(y,spin,epsilon):
    ''' Defined using the 'function' vairable from the preceding cell.
    
        *Dimensionless effective potential.*
    '''
    
    com_term = common(y,spin,epsilon)# (9*epsilon**2*spin**2)/y**8 - 6*epsilon/y**3 + 16*epsilon/y**4 + 4/y 
    if com_term !=None:
        f = y**2* ( (9*epsilon**3*spin**2*y - 30*epsilon**3*spin**2 + 9*epsilon**2*spin**2*y**4 - 42*epsilon**2*spin**2*y**3 \
                       - 4*epsilon**2*spin*y**4*numpy.sqrt(com_term)
                       + 9*epsilon**2*y**6 - 48*epsilon**2*y**5 + 64*epsilon**2*y**4 - 12*epsilon*spin**2*y**6 
                       - 8*epsilon*spin*y**7*numpy.sqrt(com_term) 
                       - epsilon*y**9*(com_term)
                       + 2*epsilon*y**8*(com_term) 
                       - 12*epsilon*y**8 + 32*epsilon*y**7 
                       - 4*spin*y**10*numpy.sqrt(com_term) 
                       - y**12 * (com_term) 
                       + 2*y**11 * (com_term) 
                       + 4*y**10 
                      )) 
        return f 
    return None 

def func_der(y,spin,epsilon): 
    ''' Defined using the 'function_derivative' variable from the preceding cell.
        
        *Dimensionless derivative of effective potential.*
    '''
    com_term = common(y,spin,epsilon) 
    if com_term != None: 
        fder =y*(9*epsilon**3*spin**2*y - 48*epsilon**3*spin**2 + 36*epsilon**2*spin**2*y**4 - 150*epsilon**2*spin**2*y**3 \
                   - 16*epsilon**2*spin*y**4*numpy.sqrt(com_term) 
                   + 54*epsilon**2*y**6 - 288*epsilon**2*y**5 + 384*epsilon**2*y**4 - 48*epsilon*spin**2*y**6  
                   - 32*epsilon*spin*y**7*numpy.sqrt(com_term) 
                   - epsilon*y**9 * (com_term) 
                   - 72*epsilon*y**8 + 192*epsilon*y**7  
                   - 16*spin*y**10*numpy.sqrt(com_term)  
                   - 4*y**12*(com_term)  
                   + 6*y**11*(com_term)  
                   +24*y**10 
                  ) 
        return fder 
    return None 



def IterativeFunc(ran_mx,spin,epsilon, prec):
    
    ''' Built-in root finding functions in Python fail to find the roots for
        certain values of spin and epsilon due to the fluctuations in the potential 
        function at higher values of epsilon and therefore we define an iterative 
        root finding function for our specific scenario.
    
        It takes a range between two numbers and runs itiratively to find the
        value of the radial coordinate R where the potential function (func()) and
        the derivative of the potential function (func_der()) change sign 
        simultaneously.
        
        The 'prec' argument determines the precision of the values found. 
    '''
    
    for i in range(len(ran_mx) -1):
        d = common(ran_mx[i],spin,epsilon)
        p = common(ran_mx[i+1],spin,epsilon)
        if d == None or p == None:
            return None
        else:
            if(func(ran_mx[i],spin,epsilon)*func(ran_mx[i+1],spin,epsilon) < 0.) and (func_der(ran_mx[i],spin,epsilon)*func_der(ran_mx[i+1],spin,epsilon) < 0.):
                if (((ran_mx[i+1] - ran_mx[i])/2.) <= prec):
                    b = ((ran_mx[i+1] + ran_mx[i])/2.)
                    return b

                else:
                    ran_newx = numpy.linspace(ran_mx[i],ran_mx[i+1], 1000)
                    ran_newmx = 0.5*(ran_newx[1::] + ran_newx[:-1:])
                    return IterativeFunc(ran_newmx,spin,epsilon, prec)
    return None

def LR_pos(spin,epsilon):
    
    ''' For different values of spin and epsilon, the location
        of the light ring is different.
        
        The function IterativeFunc is called for different ranges 
        of epsilon by adjust the minimum and maximum of the radial coordinate R. 
        If the range is too wide or narrow for R, the IterativeFunc misses the
        root and returns None. 
        
        The validity of IterativeFunc for different values of epsilon
        are found by trail and error.
    '''
    
    prec = 1e-8
    if  -10.<epsilon<=-1:
        ran_x = numpy.linspace(1.7,4.3, 1000)
        ran_mx = 0.5*(ran_x[1::] + ran_x[:-1:])
        Rpos = IterativeFunc(ran_mx,spin,epsilon,prec=prec)
    elif -1<epsilon<0.:
        ran_x = numpy.linspace(0.9,4.1, 1000)
        ran_mx = 0.5*(ran_x[1::] + ran_x[:-1:])
        Rpos = IterativeFunc(ran_mx,spin,epsilon,prec=prec)
    elif epsilon == 0:
        ran_x = numpy.linspace(0.9,4.1, 1000)
        ran_mx = 0.5*(ran_x[1::] + ran_x[:-1:])
        Rpos = IterativeFunc(ran_mx,spin,epsilon,prec=1e-1)
    elif 0.<epsilon<=75.:
        ran_x = numpy.linspace(1.5,4.0, 1000)
        ran_mx = 0.5*(ran_x[1::] + ran_x[:-1:])
        Rpos = IterativeFunc(ran_mx,spin,epsilon,prec=prec)
    elif 75.<epsilon<=100.:
        ran_x = numpy.linspace(1.4,3.45, 1000)
        ran_mx = 0.5*(ran_x[1::] + ran_x[:-1:])
        Rpos = IterativeFunc(ran_mx,spin,epsilon,prec=prec)
    elif 100.< epsilon <=160.:
        ran_x = numpy.linspace(1.5,3.2, 1000)
        ran_mx = 0.5*(ran_x[1::] + ran_x[:-1:])
        Rpos = IterativeFunc(ran_mx,spin,epsilon,prec=prec)
    elif 160.< epsilon <=280.:
        ran_x = numpy.linspace(1.4,3.1, 1000)
        ran_mx = 0.5*(ran_x[1::] + ran_x[:-1:])
        Rpos = IterativeFunc(ran_mx,spin,epsilon,prec=prec)
    elif 280.< epsilon <=300.:
        ran_x = numpy.linspace(1.4,3.0, 1000)
        ran_mx = 0.5*(ran_x[1::] + ran_x[:-1:])
        Rpos = IterativeFunc(ran_mx,spin,epsilon,prec=prec)
    else:
        ran_x = numpy.linspace(3.,10, 1000)
        ran_mx = 0.5*(ran_x[1::] + ran_x[:-1:])
        Rpos = IterativeFunc(ran_mx,spin,epsilon,prec=prec)
    return Rpos



def freqM_dim_less(spin,epsilon): 
    
    ''' Once the light ring location given the value of spin and epsilon 
        is found, we evaluate the dimensionless frequency at the light ring of a
        given value of spin and epsilon.
        
        If LR_pos() is not None, then the function finds the dimensionless frequency 
        for the given parameters. 
        If LR_pos() is None, the function returns a large number (O(10^2)) to ensure that 
        the output is a floating point and far away from the actual order of dimensionless
        frequencies (typically O(10^-1)).
        
        This expression is obtained using 'Omega1' from the preceding cell. 
    '''
    
    y = LR_pos(spin,epsilon)
    if y != None:
        com_term = common(y,spin,epsilon) #(9*epsilon**2*spin**2)/y**8 - 6*epsilon/y**3 + 16*epsilon/y**4 + 4/y 
        frequ = 2*((-3*epsilon*y + 8*epsilon + 2*y**3)/(8*epsilon*spin + 2*spin*y**3 + y**5*numpy.sqrt(com_term)))
        
        return frequ 
    return 100


def gamma0(spin,epsilon): 
    
    ''' Once the light ring location given the value of spin and epsilon 
        is found, we evaluate the inverse of dimensionless damping time at the light ring of a
        given the value of spin and epsilon. (The negative sign is conventional.)
        
        Due to the presence of a square root in the expression, there can once again
        exist complex numbers and python raises RuntimeError. We check this by splitting the 
        fraction expression and check if the ratio is positive. 
        
        If it is positive, then this function finds the inverseof dimensionless
        damping time for the given parameters.
        If not, the function returns a large number (O(-10^1)) to ensure that 
        the output is a floating point and far away from the actual order of inverse
        of dimensionless damping time (typically O(-10^-1)).
        
        If LR_pos() is not None, then this function finds the inverse of dimensionless
        damping time for the given parameters. 
        If LR_pos() is None, the function returns a large number (O(10^1)) to ensure that 
        the output is a floating point and far away from the actual order of inverse
        of dimensionless damping time (typically O(10^-1)).
        
        This expression is obtained using 'gamma1' from the preceding cell. 
    '''
    
    y = LR_pos(spin,epsilon)
    if y != None:
        com_term = common(y,spin,epsilon)#(9*epsilon**2*spin**2)/y**8 - 6*epsilon/y**3 + 16*epsilon/y**4 + 4/y 
    #     y = LR_pos(spin,epsilon)
    #     if com_term != None: 
        numerator = (54*epsilon**5*spin**6*y - 108*epsilon**5*spin**6 + 108*epsilon**4*spin**6*y**4- 270*epsilon**4*spin**6*y**3\
                    + 24*epsilon**4*spin**5*y**4*numpy.sqrt(com_term) 
                    + 117*epsilon**4*spin**4*y**6 - 480*epsilon**4*spin**4*y**5  
                    + 496*epsilon**4*spin**4*y**4 + 54*epsilon**3*spin**6*y**7 - 216*epsilon**3*spin**6*y**6  
                    + 36*epsilon**3*spin**5*y**7*numpy.sqrt(com_term) 
                    -  6*epsilon**3*spin**4*y**9*(com_term) + 126*epsilon**3*spin**4*y**9  
                    + 20*epsilon**3*spin**4*y**8*(com_term) 
                    - 648*epsilon**3*spin**4*y**8 + 808*epsilon**3*spin**4*y**7  
                    + 48*epsilon**3*spin**3*y**9*numpy.sqrt(com_term)  
                    - 96*epsilon**3*spin**3*y**8*numpy.sqrt(com_term) 
                    + 72*epsilon**3*spin**2*y**11 - 456*epsilon**3*spin**2*y**10 + 968*epsilon**3*spin**2*y**9  
                    - 688*epsilon**3*spin**2*y**8 - 54*epsilon**2*spin**6*y**9  
                    - 12*epsilon**2*spin**4*y**12*(com_term)+ 9*epsilon**2*spin**4*y**12 
                    + 42*epsilon**2*spin**4*y**11*(com_term)  
                    - 180*epsilon**2*spin**4*y**11 + 348*epsilon**2*spin**4*y**10  
                    + 24*epsilon**2*spin**3*y**12*numpy.sqrt(com_term)  
                    - 48*epsilon**2*spin**3*y**11*numpy.sqrt(com_term)  
                    - 12*epsilon**2*spin**2*y**14*(com_term) + 18*epsilon**2*spin**2*y**14 
                    + 64*epsilon**2*spin**2*y**13*(com_term) - 210*epsilon**2*spin**2*y**13  
                    - 80*epsilon**2*spin**2*y**12*(com_term) + 648*epsilon**2*spin**2*y**12  
                    - 600*epsilon**2*spin**2*y**11 + 24*epsilon**2*spin*y**14*numpy.sqrt(com_term)  
                    - 96*epsilon**2*spin*y**13*numpy.sqrt(com_term)  
                    + 96*epsilon**2*spin*y**12*numpy.sqrt(com_term)  
                    + 9*epsilon**2*y**16 - 84*epsilon**2*y**15 + 292*epsilon**2*y**14  
                    - 448*epsilon**2*y**13 + 256*epsilon**2*y**12  
                    - 12*epsilon*spin**5*y**13*numpy.sqrt(com_term)  
                    - 6*epsilon*spin**4*y**15* (com_term)  
                    + 24*epsilon*spin**4*y**14*(com_term) - 12*epsilon*spin**4*y**14 
                    + 40*epsilon*spin**4*y**13 - 24*epsilon*spin**3*y**15*numpy.sqrt(com_term) 
                    + 48*epsilon*spin**3*y**14*numpy.sqrt(com_term)  
                    - 12*epsilon*spin**2*y**17*(com_term)  
                    + 68*epsilon*spin**2*y**16*(com_term) - 24*epsilon*spin**2*y**16 
                    - 88*epsilon*spin**2*y**15*(com_term) + 120*epsilon*spin**2*y**15  
                    - 144*epsilon*spin**2*y**14 - 12*epsilon*spin*y**17*numpy.sqrt(com_term) 
                    + 48*epsilon*spin*y**16*numpy.sqrt(com_term) 
                    - 48*epsilon*spin*y**15*numpy.sqrt(com_term)  
                    - 6*epsilon*y**19*   (com_term)  
                    + 44*epsilon*y**18*  (com_term) - 12*epsilon*y**18  
                    - 104*epsilon*y**17* (com_term) + 80*epsilon*y**17 
                    +  80*epsilon*y**16* (com_term) - 176*epsilon*y**16 + 128*epsilon*y**15 
                    + 2*spin**4*y**17* (com_term) + 4*spin**4*y**16  
                    + 4*spin**2*y**19* (com_term)  
                    - 8*spin**2*y**18* (com_term) + 8*spin**2*y**18- 16*spin**2*y**17  
                    + 2*y**21*(com_term)  
                    - 8*y**20*(com_term) + 4*y**20  
                    + 8*y**19*(com_term) - 16*y**19 + 16*y**18) 
        denominator = (y**6*(4*epsilon**4*spin**2 + 16*epsilon**3*spin**2*y**3  
                            + 4*epsilon**3*spin*y**5*numpy.sqrt(com_term)  
                            - 8*epsilon**3*spin*y**4*numpy.sqrt(com_term)+ 24*epsilon**2*spin**2*y**6  
                            + 12*epsilon**2*spin*y**8*numpy.sqrt(com_term)  
                            - 24*epsilon**2*spin*y**7*numpy.sqrt(com_term)  
                            + epsilon**2*y**10*(com_term) 
                            - 4*epsilon**2*y**9*(com_term) 
                            + 4*epsilon**2*y**8*(com_term) + 16*epsilon*spin**2*y**9 
                            + 12*epsilon*spin*y**11*numpy.sqrt(com_term) 
                            - 24*epsilon*spin*y**10*numpy.sqrt(com_term)  
                            + 2*epsilon*y**13*(com_term) 
                            - 8*epsilon*y**12*(com_term)  
                            + 8*epsilon*y**11*(com_term) + 4*spin**2*y**12  
                            + 4*spin*y**14*numpy.sqrt(com_term)  
                            - 8*spin*y**13*numpy.sqrt(com_term)  
                            + y**16*(com_term)  
                            - 4*y**15*(com_term)  
                            + 4*y**14*(com_term))) 

    #         print(com_term,numerator,denominator,np.sqrt(numerator/denominator)) 
        if (numerator/denominator) >= 0: 

            gam = freqM_dim_less(spin,epsilon)*numpy.sqrt(numerator/denominator) 
            
            return -1*(gam/4)
        return -10 
    return -10

def frequency_in_hertz(mass, spin, epsilon, l, m, n): 
    
    ''' Having found the dimensionless frequency, we convert the number into SI units of Hertz
        by dividing the dimensionless frequency by the Mass in seconds. (M*lal.MTSUN_SI)
        
        As described by Glampedakis et al., QNM template of a non-Kerr scenrio is comparable
        when the numerical fits of Kerr are kept intact. To this end, Glampedakis et al. use
        an inspired function of the form of the interatomic Buckingham potential (Beta_K(x)) for the 
        numerical fits. The real part of which corresponds to the dimensionless frequency of the light ring.
        
        The quantities l, m and n are the harmonics involved in the eikonal approximation.
    '''
    
    if (freqM_dim_less(spin,epsilon)) == None:
        return None
    return m*(freqM_dim_less(spin,epsilon)+ real_beta(spin,l))/(4*numpy.pi*mass*lal.MTSUN_SI) 

def damping_in_seconds(mass, spin, epsilon, l, m, n): 
    
    ''' Having found the inverse of dimensionless damping time, we convert the number into SI units of seconds
        by multiplying 1/gamma0(spin,epsilon) by the -1 times 
        the Mass in seconds. (- M*lal.MTSUN_SI)
        
        As described by Glampedakis et al., QNM template of a non-Kerr scenrio is comparable
        when the numerical fits of Kerr are kept intact. To this end, Glampedakis et al. use
        an inspired function of the form of the interatomic Buckingham potential (Beta_K(x)) for the 
        numerical fits. The imaginary part of which corresponds to the inverse of dimensionless damping time
        of the light ring.
    '''
    
    if (gamma0(spin,epsilon)) == None:
        return None
    return -1*(mass*lal.MTSUN_SI)/((gamma0(spin,epsilon)+im_beta(spin,l)))
def real_beta(spin,l):
    
    ''' Fitting coefficients for the real part of Beta_K(x) for different l=m modes.
        (Glampedakis et al.)
    '''
    
    if   l == 2:
        a1,a2,a3,a4,a5,a6,err = 0.1282,0.4178,0.6711,0.5037,1.8331,0.7596,0.023
    elif l == 3:
        a1,a2,a3,a4,a5,a6,err = 0.1801, 0.5007,0.7064,0.5704,1.4690,0.7302, 0.005
    elif l == 4:
        a1,a2,a3,a4,a5,a6,err = 0.1974, 0.4982, 0.6808, 0.5958,1.4380, 0.7102,0.011
    elif l == 5:
        a1,a2,a3,a4,a5,a6,err = 0.2083,0.4762, 0.6524, 0.6167, 1.4615, 0.6937,0.016
    elif l == 6:
        a1,a2,a3,a4,a5,a6,err = 0.2167, 0.4458, 0.6235, 0.6373, 1.5103, 0.6791,0.021
    elif l == 7:
        a1,a2,a3,a4,a5,a6,err =0.2234, 0.4116, 0.5933, 0.6576, 1.5762, 0.6638,0.025
        
    return a1 + a2*numpy.exp(-a3*(1-(spin))**a4) - (1/(a5 + (1-(spin))**a6)) + (err*1e-2)


def im_beta(spin,l):
    
    ''' Fitting coefficients for the imaginary part of Beta_K(x) for different l=m modes.
        (Glampedakis et al.)
    '''
    
    if   l == 2:
        a1,a2,a3,a4,a5,a6,err = 0.1381,0.3131,0.5531,0.8492,2.2159,0.8544,0.004
    elif l == 3:
        a1,a2,a3,a4,a5,a6,err = 0.1590,0.3706,0.6643,0.6460,1.8889,0.6676,0.008
    elif l == 4:
        a1,a2,a3,a4,a5,a6,err = 0.1575,0.3478,0.6577,0.5840,1.9799,0.6032,0.009
    elif l == 5:
        a1,a2,a3,a4,a5,a6,err = 0.1225,0.1993,0.4855,0.6313,3.1018,0.6150,1.335
    elif l == 6:
        a1,a2,a3,a4,a5,a6,err = 0.1280,0.1947,0.5081,0.6556,3.0960,0.6434,0.665
    elif l == 7:
        a1,a2,a3,a4,a5,a6,err = -15.333,15.482,0.0011,0.3347,6.6258,0.2974,0.874

    return a1 + a2*numpy.exp(-a3*(1-(spin))**a4) - (1/(a5 + (1-(spin))**a6)) + (err*1e-2)

#### The following code chunk can be adjusted according the event and your local specifications #####
def process_iteration(i): 
    xs = spin[i] 
    epsi = e3[i] 
    mAs = mass[i] 
    result = []

    X1, Y1 = frequency_in_hertz(mAs, xs, epsi, 2, 2, 0), damping_in_seconds(mAs, xs, epsi, 2, 2, 0) 
    a = [mAs, xs, epsi] 
    data1 = numpy.array((X1, Y1)) 
    Z1 = kernel15.evaluate(data1)[0]
    for _ in range(int(Z1 * 100)): 
        result.append(a)
    return result 

if __name__ == "__main__": 
    fp = io.loadfile('your_data_file_location', 'r') 
    Samples = fp.read_samples(fp['samples'].keys()) 
    #mass_est = samples['final_mass_from_f0_tau(f_220, tau_220, 2, 2)'] 
    #spin_est = samples['final_spin_from_f0_tau(f_220, tau_220, 2, 2)'] 
    Xf15 = Samples['f_220'] 
    Yf15 = Samples['tau_220'] 
    xmin = numpy.min(Xf15) 
    xmax = numpy.max(Xf15) 

    ymin = numpy.min(Yf15) 
    ymax = numpy.max(Yf15) 

    data15 = numpy.array((Xf15,Yf15)) 

    X15, Y15= numpy.mgrid[xmin:xmax:100j, ymin:ymax:100j] 


    positions15 = numpy.vstack([X15.ravel(), Y15.ravel()]) 

    kernel15 = stats.gaussian_kde(data15) 
    
    num_processes = mp.cpu_count() 
    size =  int(3*num_processes**3) ##ADJUST ACCORDING TO YOUR CPU SPECFICS

    mass = numpy.random.uniform(low = (1-0.50)*67.6, high = (1+0.50)*67.6, size = int(size)) ## from IMR of GW150914
    spin = numpy.random.uniform(low = 0., high = 1.0, size = int(size))  
    e3 = numpy.random.uniform(low = -30.0, high = 300.0, size = int(size))

    

    with mp.Pool(processes=num_processes) as pool: 
        results = list(tqdm(pool.imap(process_iteration, range(size)), total=size)) 

    # Flatten the results 
    post15 = [item for sublist in results for item in sublist] 

## Extracting values
Mvals15=[]
e3vals15=[]
Xvals15=[]

for i in range(len(post15)):
    Mvals15.append(post15[i][0])
    Xvals15.append(post15[i][1])
    e3vals15.append(post15[i][2])
## Saving GW150914 files
file_name = "your_file_name_and_save_location"
with h5py.File(file_name, 'w') as f:

    f.create_dataset('All_param', data = post15)
    f.create_dataset('Mass', data=Mvals15)
    f.create_dataset('Spin', data=Xvals15)
    f.create_dataset('Epsilon',data = e3vals15)
