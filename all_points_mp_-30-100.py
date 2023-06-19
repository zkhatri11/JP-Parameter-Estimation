#!/usr/bin/env python
# coding: utf-8



# In[1]:
with open('output.txt', 'w') as f:
    print('Starting script...')
    # rest of your code here
    

    import numpy as np
    import lal
    import random
    from pycbc.inference import io
    import h5py
    import time
    import multiprocessing as mp


# In[2]:


    fp = io.loadfile('GW150914_3ms_freq_tau_4knliv.hdf', 'r')
    samples = fp.read_samples(fp['samples'].keys())
    mass_est = samples['final_mass_from_f0_tau(f_220, tau_220, 2, 2)']
    spin_est = samples['final_spin_from_f0_tau(f_220, tau_220, 2, 2)']
    Freq_data= samples['f_220']
    Tau_data = samples['tau_220']


# In[3]:


    f1 = 1.5251
    f2 = -1.1568
    f3 = 0.1292

    q1 = 0.7000
    q2 = 1.4187
    q3 = -0.4990

#c = sp.speed_of_light
#g = sp.gravitational_constant

    c=1
    g=1
    # expressions for calculating frequency and damping in JP geometry
    
    def f_JP(m,x,e):
        return 2*(((f1+f2*((1.-x)**f3))*c**2)/(2.*np.pi*m*g) + (e*((1/(81.*np.sqrt(3))+((10.*x)/(729.))+((47.*x**2)/(1458.*np.sqrt(3))))))/(2.*np.pi*m))/lal.MTSUN_SI

    def t_JP(m,x,e):
        return (((c**2/(2.*np.pi*m*g))*(((f1 + f2*((1.-x)**f3))*np.pi)/(q1 + q2*((1-x)**q3)) - e*((x)/(486.) + (16.*x**2)/(2187.*np.sqrt(3))))/lal.MTSUN_SI))**(-1)

    #compute the mass for a given frequency/damping time data value, spin and epsilon

    def mfind_tau(t, x, e):
        return (t_JP((t)**(-1),x,e))**(-1)

    def mfind_f(f,x,e):
        return (f_JP(f,x,e))



    # In[101]:

    # epsilon and spin ranges
    n_chi=1000
    n_eps = 1000
    
    chi_vals1 = np.linspace(0, 0.99, n_chi)
    e_vals1 = np.linspace(-30., 100., n_eps)


    # In[103]:





    f_tolerance = 0.0001
    m_tolerance = 0.001
    data_pts = len(Freq_data)
    m_JP = []
    chi_JP = []
    eps_JP = []
    freq_JP = []
    tau_JP = []
    
    def process_chunk(ind, chi_vals1, e_vals1, queue):
        results = []
        min_m = 20
        max_m = 200

        for i in ind:
            for x in chi_vals1:
                for e in e_vals1:
                    temp = mfind_tau(Tau_data[i], x, e)
                    Temp = mfind_f(Freq_data[i], x, e)
                    diff_m = abs(temp-Temp)
                    if diff_m <= m_tolerance:
                        if min_m <= temp <= max_m:
                            diff_f = np.abs(f_JP(temp, x, e)-Freq_data[i])
                            if diff_f <= f_tolerance:
                                results.append((temp, x, e, f_JP(temp, x, e), t_JP(temp, x, e)))

        queue.put(results)
    start = time.time()
    
    # Multiprocessing code for reducing computation time
    if __name__ == '__main__':

        ind = np.random.choice(range(0, len(Freq_data)), data_pts, replace=False)
        chunksize = int(len(ind) / (4 * mp.cpu_count()))

        with mp.Pool() as pool:
            queue = mp.Manager().Queue()
            for i in range(4 * mp.cpu_count()):
                print('Script running', i+1,'/',(4 * mp.cpu_count()))
                start_idx = i * chunksize
                end_idx = start_idx + chunksize if i < (4 * mp.cpu_count() - 1) else len(ind)
                pool.apply_async(process_chunk, args=(ind[start_idx:end_idx], chi_vals1, e_vals1, queue))

            pool.close()
            pool.join()

            while not queue.empty():
                results = queue.get()
                for result in results:
                    m_JP.append(result[0])
                    chi_JP.append(result[1])
                    eps_JP.append(result[2])
                    freq_JP.append(result[3])
                    tau_JP.append(result[4])
    end = time.time()
    


    # In[104]:


    with h5py.File('final_posteriors_tot_epsprior_-30-100.hdf', 'w') as f:
        f.create_dataset('freq_JP', data=freq_JP)
        f.create_dataset('tau_JP', data=tau_JP)
        f.create_dataset('Mass', data=m_JP)
        f.create_dataset('Spin', data=chi_JP)
        f.create_dataset('Epsilon', data=eps_JP)

    print('Script complete!')
