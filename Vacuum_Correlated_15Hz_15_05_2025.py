#corellated noise 
import numpy as np
from qutip import*
import matplotlib.pyplot as plt
import math
import cmath 
import random
import csv
pi=np.pi
n=200
N=20
ntraj = 1
meanw1 =  1.34e6
meanw2 =  1.47e6
std_dev = 15#15*2*np.pi #sigma2 = sigma1*w2/w1
#delta = 2 #x*ratio of frequency 
num_samples = 20
zero_mean_samples = np.random.normal(loc=0.0, scale=std_dev, size=num_samples)
w1_values = 2*np.pi*( meanw1 + zero_mean_samples)
w2_values= 2*np.pi*(meanw2 + zero_mean_samples*(meanw2/meanw1)  )#mean2 +  
R1=[]
R2=[]
R3=[]
C1=[]
C2=[]
C3=[]
chi=[]


for i in range((num_samples)):
# defining variables required for the hamiltonian 
        w1 = w1_values[i]
        w2 = w2_values[i]
        #print(w1,w2)
        alpha1 = 0
        alpha2 = 0

        # Qubit and oscillator operators
        a1  = tensor(qeye(2),destroy(N),qeye(N))  # Oscillator  annihilation operator and identity on qubit
        a2  = tensor(qeye(2),qeye(N),destroy(N)) 
        sm = tensor( (basis(2,1)*basis(2,1).dag()) ,qeye(N),qeye(N) ) #Excited state projection for qubit and identity on mode

        # Identity operators for qubit and oscillator
        I_q = qeye(2)
        I_o = qeye(N)
        #sm_q=basis(2,1)*basis(2,1).dag() #just the excited state projection for the qubit
        sm_q = sigmaz()
        # free hamiltonian of the oscillators 
        H_0 = w1*a1.dag()*a1 + w2*a2.dag()*a2
            
        # interaction hamiltonian 
        H_int11 = tensor(sm_q, destroy(N).dag(),qeye(N))
        H_int21 = tensor(sm_q,destroy(N), qeye(N))
        H_int12 = tensor(sm_q, qeye(N), destroy(N).dag())
        H_int22 = tensor(sm_q, qeye(N), destroy(N))
        proj_ground= tensor(fock_dm(2,0), qeye(N),qeye(N))
        proj_excited = tensor(fock_dm(2,1),qeye(N),qeye(N))

        def R_phi(phi):
            # Define the matrix elements based on the given formula
            matrix1 = (1 / np.sqrt(2)) * np.array([[1, np.exp(1j * phi)],
                                                [-np.exp(-1j *phi), 1]])
            matrix = tensor(Qobj(matrix1),qeye(N),qeye(N))
            # Return as a QuTiP quantum object
            return Qobj(matrix)

        def interaction_diftime(w1,w2,lam1,lam2,d1,d2,ini_state,t_in1,t_in2,tau):
            temp1 = t_in1 + tau
            temp2 = t_in2 + tau
                # Define the functions for the time-dependent terms
            def H_int11_func(t):
                return lam1 * np.exp(-1j * (meanw1 - d1) * t) * np.exp(1j * w1 * t)

            def H_int21_func(t):
                return lam1.conjugate() * np.exp(1j * (meanw1 - d1) * t) * np.exp(-1j * w1 * t)

            def H_int12_func(t):
                return lam2 * np.exp(-1j * (meanw2 - d2) * t) * np.exp(1j * w2 * t)

            def H_int22_func(t):
                return lam2.conjugate() * np.exp(1j * (meanw2 - d2) * t) * np.exp(-1j * w2 * t)

            # Now create the QobjEvo objects using these functions
            H = QobjEvo([
                [H_int11, H_int11_func],
                [H_int21, H_int21_func],
                [H_int12, H_int12_func],
                [H_int22, H_int22_func]
            ])

            H1 = QobjEvo([
                [H_int11, H_int11_func],
                [H_int21, H_int21_func]
            ])

            H2 = QobjEvo([
                [H_int12, H_int12_func],
                [H_int22, H_int22_func]
            ])

            if t_in1==t_in2:
                temp = temp1
                t= np.linspace(t_in1,temp,n)
                fin_state = mesolve(H,ini_state,t)
                final_state_avg = fin_state.final_state
            return final_state_avg



        def ramsey(phi,w1,w2,lam1,lam2,d1,d2,ini_state,t_in1,t_in2,tau,dm1,dm2):
            
            state_after_1pulse = R_phi(phi)*ini_state
            
            interaction_time = interaction_diftime(w1,w2,lam1,lam2,d1,d2,state_after_1pulse,t_in1,t_in2,tau)
            
            displace_state = tensor(
            qeye(2),
            displace(N, np.sign(dm1) * 0.67 * np.exp(-1j * angle * 2 +  1j * (2 * np.pi * (abs(dm1) - 1) / 3))),
            displace(N, np.sign(dm2) * 0.67 * np.exp(-1j * angle * 2 +  1j * (2 * np.pi * (abs(dm2) - 1) / 3)))
        ) * interaction_time
            

            second_pulse = R_phi(0)*displace_state
            return second_pulse

        def measure(state):
            p_plus = expect(proj_excited,state)
            p_minus = expect(proj_ground,state)
            state_e = (1/np.sqrt(p_plus))*proj_excited*state
            state_g = (1/np.sqrt(p_minus))*proj_ground*state
            return p_plus,p_minus,state_e,state_g

        x=Qobj(np.array([[0, 1],[1, 0]]))
        notgate=tensor(x,qeye(N),qeye(N))
        t=110.67*10**-6
        d=3.5*2*pi*10**3
        l=2.5*10**3*2*pi#18.238*2*pi*10**3
        t_b = 600*10**-6
        l=l/2
        angle = (l**2/d**2)*(d*t - np.sin(d*t))/2


        #row1

        r1t1 = 0 
        r1t2 = (2*pi*math.ceil((t_b+r1t1+t)/(2*pi/d))/d)
        r1t3 = pi/d + (2*pi*math.ceil((t_b+r1t2+t)/(2*pi/d))/d)

        # #row2 
        r2t1 = 0
        r2t2 = (2*pi*math.ceil((t_b+r2t1+t)/(2*pi/d))/d)
        r2t3 =pi/d + (2*pi*math.ceil((t_b+r2t2+t)/(2*pi/d))/d)

        r3t1 = 0
        r3t2 = (2*pi*(math.ceil((t_b+r3t1+t)/(2*pi/d)))/d)
        r3t3 = (2*pi*(math.ceil((t_b+r3t2+t)/(2*pi/d)))/d)


        #If the columns have to start at the same time

        c1t1= c2t1 = 0
        c1t2 = c2t2 = (2*pi*math.ceil((t_b+c1t1+t_b)/(2*pi/d))/d)
        c1t3 = c2t3 = (2*pi*math.ceil((t_b+c1t2+t_b)/(2*pi/d))/d)

        c3t1 = pi/d
        c3t2 = 5*pi/(3*d) + (2*pi*math.ceil((t_b+c3t1+t)/(2*pi/d))/d)
        c3t3 = 7*pi/(3*d) + (2*pi*math.ceil((t_b+c3t2+t)/(2*pi/d))/d)


        alpha1 = 0
        alpha2 = 0
        # Define the coherent states for the oscillator
        coherent_state1 = coherent(N, alpha1)  # Coherent state for the oscillator here at vacuum
        coherent_state2 = coherent(N,alpha2)

        

        # # Combine the states using the tensor product (qubit, mode 1, mode 2)
        initial_state = tensor(basis(2,0),coherent_state1,coherent_state2)


        def row1(r1t11,r1t12,r1t21,r1t22,r1t31,r1t32):
            # Now we find the expectation value of the first term in the expresion (A11R A12R A12R) in Eq. 10 of the paper 10.1103/PhysRevLett.114.250403
            #print(r1t11,r1t12,r1t21,r1t22,r1t31,r1t32)
            angle = (l**2/d**2)*(d*t - np.sin(d*t))/2


            R111 = ramsey(angle,w1,w2,l,0,d,d,initial_state,r1t11,r1t12,t,-1,0)
            #R111 = FreeEvolution(R111,r1t11 + t, r1t21)
            R111_prob = measure(R111) # +,- prob(t1 =+),Prob(t1=-)


            R112 = ramsey(angle,w1,w2,0,l,d,d,notgate*R111_prob[2],r1t21,r1t22,t,0,-1)
            #R112 = FreeEvolution(R112,r1t21 + t, r1t31)
            R112_prob = measure(R112) # prob(t2=+|t1=+) prob(t2=-|t1=+)

            R113 = ramsey(angle*2,w1,w2,l,l,d,d,notgate*R112_prob[2],r1t31,r1t32,t,1,1)
            R113_prob = measure(R113) # prob(t3=+|t2=+|t1=+) prob(t3=-|t2=+|t1=+)

            R114 = ramsey(angle*2,w1,w2,l,l,d,d,R112_prob[3],r1t31,r1t32,t,1,1)
            R114_prob = measure(R114) #  prob(t3=+|t2=-|t1=+) prob(t3=-|t2=-|t1=+)

            R115 = ramsey(angle,w1,w2,0,l,d,d,R111_prob[3],r1t21,r1t22,t,0,-1)
            #R115 =  FreeEvolution(R115,r1t21 + t, r1t31)
            R115_prob = measure(R115)  #prob(t2=+|t1=-) prob(t2=-|t1=-)

            R116 = ramsey(angle*2,w1,w2,l,l,d,d, notgate*R115_prob[2],r1t31,r1t32,t,1,1)
            R116_prob = measure(R116) # prob(t3=+|t2=+|t1=-) prob(t3=-|t2=+|t1=-)

            R117 = ramsey(angle*2,w1,w2,l,l,d,d,R115_prob[3],r1t31,r1t32,t,1,1)
            R117_prob = measure(R117) # prob(t3=+|t2=-|t1=-) prob(t3=-|t2=-|t1=-)


            # Now we find the expectation value of the first term in the expresion (A11I A12I A12R) in Eq. 10 of the paper 10.1103/PhysRevLett.114.250403



            R121 = ramsey(angle+pi/2,w1,w2,l,0,d,d,initial_state,r1t11,r1t12,t,-1,0)
            #R121 = FreeEvolution(R121,r1t11 + t, r1t21)
            R121_prob = measure(R121) # +,- prob(t1 =+),Prob(t1=-)


            R122 = ramsey(angle+pi/2,w1,w2,0,l,d,d,notgate*R121_prob[2],r1t21,r1t22,t,0,-1)
            #R122 = FreeEvolution(R122,r1t21 + t, r1t31)
            R122_prob = measure(R122) # prob(t2=+|t1=+) prob(t2=-|t1=+)

            R123 = ramsey(angle*2,w1,w2,l,l,d,d,notgate*R122_prob[2],r1t31,r1t32,t,1,1)
            R123_prob = measure(R123) # prob(t3=+|t2=+|t1=+) prob(t3=-|t2=+|t1=+)

            R124 = ramsey(angle*2,w1,w2,l,l,d,d,R122_prob[3],r1t31,r1t32,t,1,1)
            R124_prob = measure(R124) #  prob(t3=+|t2=-|t1=+) prob(t3=-|t2=-|t1=+)

            R125 = ramsey(angle+pi/2,w1,w2,0,l,d,d,R121_prob[3],r1t21,r1t22,t,0,-1)
            #R125 = FreeEvolution(R125,r1t21 + t, r1t31)
            R125_prob = measure(R125)  #prob(t2=+|t1=-) prob(t2=-|t1=-)

            R126 = ramsey(angle*2,w1,w2,l,l,d,d, notgate*R125_prob[2],r1t31,r1t32,t,1,1)
            R126_prob = measure(R126) # prob(t3=+|t2=+|t1=-) prob(t3=-|t2=+|t1=-)

            R127 = ramsey(angle*2,w1,w2,l,l,d,d,R125_prob[3],r1t31,r1t32,t,1,1)
            R127_prob = measure(R127) # prob(t3=+|t2=-|t1=-) prob(t3=-|t2=-|t1=-)

            # Now we find the expectation value of the first term in the expresion (A11I A12R A12I) in Eq. 10 of the paper 10.1103/PhysRevLett.114.250403



            R131 = ramsey(angle+pi/2,w1,w2,l,0,d,d,initial_state,r1t11,r1t12,t,-1,0)
            #R131 = FreeEvolution(R131,r1t11 + t, r1t21)
            R131_prob = measure(R131) # +,- prob(t1 =+),Prob(t1=-)


            R132 = ramsey(angle,w1,w2,0,l,d,d,notgate*R131_prob[2],r1t21,r1t22,t,0,-1)
            #R132 = FreeEvolution(R132,r1t21 + t, r1t31)
            R132_prob = measure(R132) # prob(t2=+|t1=+) prob(t2=-|t1=+)

            R133 = ramsey(angle*2+pi/2,w1,w2,l,l,d,d,notgate*R132_prob[2],r1t31,r1t32,t,1,1)
            R133_prob = measure(R133) # prob(t3=+|t2=+|t1=+) prob(t3=-|t2=+|t1=+)

            R134 = ramsey(angle*2+pi/2,w1,w2,l,l,d,d,R132_prob[3],r1t31,r1t32,t,1,1)
            R134_prob = measure(R134) #  prob(t3=+|t2=-|t1=+) prob(t3=-|t2=-|t1=+)

            R135 = ramsey(angle,w1,w2,0,l,d,d,R131_prob[3],r1t21,r1t22,t,0,-1)
            #R135 = FreeEvolution(R135,r1t21 + t, r1t31)
            R135_prob = measure(R135)  #prob(t2=+|t1=-) prob(t2=-|t1=-)

            R136 = ramsey(angle*2+pi/2,w1,w2,l,l,d,d, notgate*R135_prob[2],r1t31,r1t32,t,1,1)
            R136_prob = measure(R136) # prob(t3=+|t2=+|t1=-) prob(t3=-|t2=+|t1=-)

            R137 = ramsey(angle*2+pi/2,w1,w2,l,l,d,d,R135_prob[3],r1t31,r1t32,t,1,1)
            R137_prob = measure(R137) # prob(t3=+|t2=-|t1=-) prob(t3=-|t2=-|t1=-)


            # Now we find the expectation value of the first term in the expresion (A11R A12I A12I) in Eq. 10 of the paper 10.1103/PhysRevLett.114.250403

            R141 = ramsey(angle,w1,w2,l,0,d,d,initial_state,r1t11,r1t12,t,-1,0)
            #R141 = FreeEvolution(R141,r1t11 + t, r1t21)
            R141_prob = measure(R141) # +,- prob(t1 =+),Prob(t1=-)


            R142 = ramsey(angle+pi/2,w1,w2,0,l,d,d,notgate*R141_prob[2],r1t21,r1t22,t,0,-1)
            #R142 = FreeEvolution(R142,r1t21 + t, r1t31)
            R142_prob = measure(R142) # prob(t2=+|t1=+) prob(t2=-|t1=+)

            R143 = ramsey(angle*2+pi/2,w1,w2,l,l,d,d,notgate*R142_prob[2],r1t31,r1t32,t,1,1)
            R143_prob = measure(R143) # prob(t3=+|t2=+|t1=+) prob(t3=-|t2=+|t1=+)

            R144 = ramsey(angle*2+pi/2,w1,w2,l,l,d,d,R142_prob[3],r1t31,r1t32,t,1,1)
            R144_prob = measure(R144) #  prob(t3=+|t2=-|t1=+) prob(t3=-|t2=-|t1=+)

            R145 = ramsey(angle+pi/2,w1,w2,0,l,d,d,R141_prob[3],r1t21,r1t22,t,0,-1)
            #R145 = FreeEvolution(R145,r1t21 + t, r1t31)
            R145_prob = measure(R145)  #prob(t2=+|t1=-) prob(t2=-|t1=-)

            R146 = ramsey(angle*2+pi/2,w1,w2,l,l,d,d, notgate*R145_prob[2],r1t31,r1t32,t,1,1)
            R146_prob = measure(R146) # prob(t3=+|t2=+|t1=-) prob(t3=-|t2=+|t1=-)

            R147 = ramsey(angle*2+pi/2,w1,w2,l,l,d,d,R145_prob[3],r1t31,r1t32,t,1,1)
            R147_prob = measure(R147) # prob(t3=+|t2=-|t1=-) prob(t3=-|t2=-|t1=-)

            #three time corellation
            expectterm1= (R111_prob[0]*R112_prob[0]*(R113_prob[0]-R113_prob[1])) + (R111_prob[0]*R112_prob[1]*(-R114_prob[0]+R114_prob[1])) + (R111_prob[1]*R115_prob[0]*(-R116_prob[0]+R116_prob[1]))+(R111_prob[1]*R115_prob[1]*(R117_prob[0]-R117_prob[1]))
            expectterm2=(R121_prob[0]*R122_prob[0]*(R123_prob[0]-R123_prob[1])) + (R121_prob[0]*R122_prob[1]*(-R124_prob[0]+R124_prob[1])) + (R121_prob[1]*R125_prob[0]*(-R126_prob[0]+R126_prob[1]))+(R121_prob[1]*R125_prob[1]*(R127_prob[0]-R127_prob[1]))
            expectterm3=(R131_prob[0]*R132_prob[0]*(R133_prob[0]-R133_prob[1])) + (R131_prob[0]*R132_prob[1]*(-R134_prob[0]+R134_prob[1])) + (R131_prob[1]*R135_prob[0]*(-R136_prob[0]+R136_prob[1]))+(R131_prob[1]*R135_prob[1]*(R137_prob[0]-R137_prob[1]))
            expectterm4=(R141_prob[0]*R142_prob[0]*(R143_prob[0]-R143_prob[1])) + (R141_prob[0]*R142_prob[1]*(-R144_prob[0]+R144_prob[1])) + (R141_prob[1]*R145_prob[0]*(-R146_prob[0]+R146_prob[1]))+(R141_prob[1]*R145_prob[1]*(R147_prob[0]-R147_prob[1]))

            total=  (expectterm1 - expectterm2 - expectterm3 - expectterm4)
            
            return total


        R1.append(row1(r1t1,r1t1,r1t2,r1t2,r1t3,r1t3))

        def row2(r1t11,r1t12,r1t21,r1t22,r1t31,r1t32):
            # Now we find the expectation value of the first term in the expresion (A11R A12R A12R) in Eq. 10 of the paper 10.1103/PhysRevLett.114.250403
            angle = (l**2/d**2)*(d*t - np.sin(d*t))/2


            R111 = ramsey(angle,w1,w2,0,np.exp(2*pi/3*1j)*l,d,d,initial_state,r1t11,r1t12,t,0,-2)
            #R111 = FreeEvolution(R111,r1t11 + t, r1t21)
            R111_prob = measure(R111) # +,- prob(t1 =+),Prob(t1=-)


            R112 = ramsey(angle,w1,w2,np.exp(2*pi/3*1j)*l,0,d,d,notgate*R111_prob[2],r1t21,r1t22,t,-2,0)
            #R112 = FreeEvolution(R112,r1t21 + t, r1t31)
            R112_prob = measure(R112) # prob(t2=+|t1=+) prob(t2=-|t1=+)

            R113 = ramsey(angle*2,w1,w2,np.exp(2*pi/3*1j)*l,np.exp(2*pi/3*1j)*l,d,d,notgate*R112_prob[2],r1t31,r1t32,t,2,2)
            R113_prob = measure(R113) # prob(t3=+|t2=+|t1=+) prob(t3=-|t2=+|t1=+)

            R114 = ramsey(angle*2,w1,w2,np.exp(2*pi/3*1j)*l,np.exp(2*pi/3*1j)*l,d,d,R112_prob[3],r1t31,r1t32,t,2,2)
            R114_prob = measure(R114) #  prob(t3=+|t2=-|t1=+) prob(t3=-|t2=-|t1=+)

            R115 = ramsey(angle,w1,w2,np.exp(2*pi/3*1j)*l,0,d,d,R111_prob[3],r1t21,r1t22,t,-2,0)
            #R115 =  FreeEvolution(R115,r1t21 + t, r1t31)
            R115_prob = measure(R115)  #prob(t2=+|t1=-) prob(t2=-|t1=-)

            R116 = ramsey(angle*2,w1,w2,np.exp(2*pi/3*1j)*l,np.exp(2*pi/3*1j)*l,d,d, notgate*R115_prob[2],r1t31,r1t32,t,2,2)
            R116_prob = measure(R116) # prob(t3=+|t2=+|t1=-) prob(t3=-|t2=+|t1=-)

            R117 = ramsey(angle*2,w1,w2,np.exp(2*pi/3*1j)*l,np.exp(2*pi/3*1j)*l,d,d,R115_prob[3],r1t31,r1t32,t,2,2)
            R117_prob = measure(R117) # prob(t3=+|t2=-|t1=-) prob(t3=-|t2=-|t1=-)


            # Now we find the expectation value of the first term in the expresion (A11I A12I A12R) in Eq. 10 of the paper 10.1103/PhysRevLett.114.250403



            R121 = ramsey(angle+pi/2,w1,w2,0,np.exp(2*pi/3*1j)*l,d,d,initial_state,r1t11,r1t12,t,0,-2)
            R121_prob = measure(R121) # +,- prob(t1 =+),Prob(t1=-)


            R122 = ramsey(angle+pi/2,w1,w2,np.exp(2*pi/3*1j)*l,0,d,d,notgate*R121_prob[2],r1t21,r1t22,t,-2,0)
            R122_prob = measure(R122) # prob(t2=+|t1=+) prob(t2=-|t1=+)

            R123 = ramsey(angle*2,w1,w2,np.exp(2*pi/3*1j)*l,np.exp(2*pi/3*1j)*l,d,d,notgate*R122_prob[2],r1t31,r1t32,t,2,2)
            R123_prob = measure(R123) # prob(t3=+|t2=+|t1=+) prob(t3=-|t2=+|t1=+)

            R124 = ramsey(angle*2,w1,w2,np.exp(2*pi/3*1j)*l,np.exp(2*pi/3*1j)*l,d,d,R122_prob[3],r1t31,r1t32,t,2,2)
            R124_prob = measure(R124) #  prob(t3=+|t2=-|t1=+) prob(t3=-|t2=-|t1=+)

            R125 = ramsey(angle+pi/2,w1,w2,np.exp(2*pi/3*1j)*l,0,d,d,R121_prob[3],r1t21,r1t22,t,-2,0)
            R125_prob = measure(R125)  #prob(t2=+|t1=-) prob(t2=-|t1=-)

            R126 = ramsey(angle*2,w1,w2,np.exp(2*pi/3*1j)*l,np.exp(2*pi/3*1j)*l,d,d, notgate*R125_prob[2],r1t31,r1t32,t,2,2)
            R126_prob = measure(R126) # prob(t3=+|t2=+|t1=-) prob(t3=-|t2=+|t1=-)

            R127 = ramsey(angle*2,w1,w2,np.exp(2*pi/3*1j)*l,np.exp(2*pi/3*1j)*l,d,d,R125_prob[3],r1t31,r1t32,t,2,2)
            R127_prob = measure(R127) # prob(t3=+|t2=-|t1=-) prob(t3=-|t2=-|t1=-)

            # Now we find the expectation value of the first term in the expresion (A11I A12R A12I) in Eq. 10 of the paper 10.1103/PhysRevLett.114.250403



            R131 = ramsey(angle+pi/2,w1,w2,0,np.exp(2*pi/3*1j)*l,d,d,initial_state,r1t11,r1t12,t,0,-2)
            R131_prob = measure(R131) # +,- prob(t1 =+),Prob(t1=-)


            R132 = ramsey(angle,w1,w2,np.exp(2*pi/3*1j)*l,0,d,d,notgate*R131_prob[2],r1t21,r1t22,t,-2,0)
            R132_prob = measure(R132) # prob(t2=+|t1=+) prob(t2=-|t1=+)

            R133 = ramsey(angle*2+pi/2,w1,w2,np.exp(2*pi/3*1j)*l,np.exp(2*pi/3*1j)*l,d,d,notgate*R132_prob[2],r1t31,r1t32,t,2,2)
            R133_prob = measure(R133) # prob(t3=+|t2=+|t1=+) prob(t3=-|t2=+|t1=+)

            R134 = ramsey(angle*2+pi/2,w1,w2,np.exp(2*pi/3*1j)*l,np.exp(2*pi/3*1j)*l,d,d,R132_prob[3],r1t31,r1t32,t,2,2)
            R134_prob = measure(R134) #  prob(t3=+|t2=-|t1=+) prob(t3=-|t2=-|t1=+)

            R135 = ramsey(angle,w1,w2,np.exp(2*pi/3*1j)*l,0,d,d,R131_prob[3],r1t21,r1t22,t,-2,0)
            R135_prob = measure(R135)  #prob(t2=+|t1=-) prob(t2=-|t1=-)

            R136 = ramsey(angle*2+pi/2,w1,w2,np.exp(2*pi/3*1j)*l,np.exp(2*pi/3*1j)*l,d,d, notgate*R135_prob[2],r1t31,r1t32,t,2,2)
            R136_prob = measure(R136) # prob(t3=+|t2=+|t1=-) prob(t3=-|t2=+|t1=-)

            R137 = ramsey(angle*2+pi/2,w1,w2,np.exp(2*pi/3*1j)*l,np.exp(2*pi/3*1j)*l,d,d,R135_prob[3],r1t31,r1t32,t,2,2)
            R137_prob = measure(R137) # prob(t3=+|t2=-|t1=-) prob(t3=-|t2=-|t1=-)


            # Now we find the expectation value of the first term in the expresion (A11R A12I A12I) in Eq. 10 of the paper 10.1103/PhysRevLett.114.250403

            R141 = ramsey(angle,w1,w2,0,np.exp(2*pi/3*1j)*l,d,d,initial_state,r1t11,r1t12,t,0,-2)
            R141_prob = measure(R141) # +,- prob(t1 =+),Prob(t1=-)


            R142 = ramsey(angle+pi/2,w1,w2,np.exp(2*pi/3*1j)*l,0,d,d,notgate*R141_prob[2],r1t21,r1t22,t,-2,0)
            R142_prob = measure(R142) # prob(t2=+|t1=+) prob(t2=-|t1=+)

            R143 = ramsey(angle*2+pi/2,w1,w2,np.exp(2*pi/3*1j)*l,np.exp(2*pi/3*1j)*l,d,d,notgate*R142_prob[2],r1t31,r1t32,t,2,2)
            R143_prob = measure(R143) # prob(t3=+|t2=+|t1=+) prob(t3=-|t2=+|t1=+)

            R144 = ramsey(angle*2+pi/2,w1,w2,np.exp(2*pi/3*1j)*l,np.exp(2*pi/3*1j)*l,d,d,R142_prob[3],r1t31,r1t32,t,2,2)
            R144_prob = measure(R144) #  prob(t3=+|t2=-|t1=+) prob(t3=-|t2=-|t1=+)

            R145 = ramsey(angle+pi/2,w1,w2,np.exp(2*pi/3*1j)*l,0,d,d,R141_prob[3],r1t21,r1t22,t,-2,0)
            R145_prob = measure(R145)  #prob(t2=+|t1=-) prob(t2=-|t1=-)

            R146 = ramsey(angle*2+pi/2,w1,w2,np.exp(2*pi/3*1j)*l,np.exp(2*pi/3*1j)*l,d,d, notgate*R145_prob[2],r1t31,r1t32,t,2,2)
            R146_prob = measure(R146) # prob(t3=+|t2=+|t1=-) prob(t3=-|t2=+|t1=-)

            R147 = ramsey(angle*2+pi/2,w1,w2,np.exp(2*pi/3*1j)*l,np.exp(2*pi/3*1j)*l,d,d,R145_prob[3],r1t31,r1t32,t,2,2)
            R147_prob = measure(R147) # prob(t3=+|t2=-|t1=-) prob(t3=-|t2=-|t1=-)

            #three time corellation
            expectterm1= (R111_prob[0]*R112_prob[0]*(R113_prob[0]-R113_prob[1])) + (R111_prob[0]*R112_prob[1]*(-R114_prob[0]+R114_prob[1])) + (R111_prob[1]*R115_prob[0]*(-R116_prob[0]+R116_prob[1]))+(R111_prob[1]*R115_prob[1]*(R117_prob[0]-R117_prob[1]))
            expectterm2=(R121_prob[0]*R122_prob[0]*(R123_prob[0]-R123_prob[1])) + (R121_prob[0]*R122_prob[1]*(-R124_prob[0]+R124_prob[1])) + (R121_prob[1]*R125_prob[0]*(-R126_prob[0]+R126_prob[1]))+(R121_prob[1]*R125_prob[1]*(R127_prob[0]-R127_prob[1]))
            expectterm3=(R131_prob[0]*R132_prob[0]*(R133_prob[0]-R133_prob[1])) + (R131_prob[0]*R132_prob[1]*(-R134_prob[0]+R134_prob[1])) + (R131_prob[1]*R135_prob[0]*(-R136_prob[0]+R136_prob[1]))+(R131_prob[1]*R135_prob[1]*(R137_prob[0]-R137_prob[1]))
            expectterm4=(R141_prob[0]*R142_prob[0]*(R143_prob[0]-R143_prob[1])) + (R141_prob[0]*R142_prob[1]*(-R144_prob[0]+R144_prob[1])) + (R141_prob[1]*R145_prob[0]*(-R146_prob[0]+R146_prob[1]))+(R141_prob[1]*R145_prob[1]*(R147_prob[0]-R147_prob[1]))

            total=  (expectterm1 - expectterm2 - expectterm3 - expectterm4)
            
            return total



        R2.append(row2(r2t1,r2t1,r2t2,r2t2,r2t3,r2t3))
        #print(np.sign(2) * 0.67 * np.exp((-1j * angle * 2) + (np.sign(2) * 1j * (2 * np.pi * (abs(2) - 1) / 3))))

        def row3(r1t11,r1t12,r1t21,r1t22,r1t31,r1t32):
            # Now we find the expectation value of the first term in the expresion (A11R A12R A12R) in Eq. 10 of the paper 10.1103/PhysRevLett.114.250403
            #print(r1t11,r1t12,r1t21,r1t22,r1t31,r1t32)
            angle = (l**2/d**2)*(d*t - np.sin(d*t))/2


            R111 = ramsey(angle*2,w1,w2,-l,-np.exp(2*pi/3*1j)*l,d,d,initial_state,r1t11,r1t12,t,1,2)
            #R111 = FreeEvolution(R111,r1t11 + t, r1t21)
            R111_prob = measure(R111) # +,- prob(t1 =+),Prob(t1=-)


            R112 = ramsey(angle*2,w1,w2,-np.exp(2*pi/3*1j)*l,-l,d,d,notgate*R111_prob[2],r1t21,r1t22,t,2,1)
            #R112 = FreeEvolution(R112,r1t21 + t, r1t31)
            R112_prob = measure(R112) # prob(t2=+|t1=+) prob(t2=-|t1=+)

            R113 = ramsey(angle*2,w1,w2,-np.exp(4*pi/3*1j)*l,-np.exp(4*pi/3*1j)*l,d,d,notgate*R112_prob[2],r1t31,r1t32,t,3,3)
            R113_prob = measure(R113) # prob(t3=+|t2=+|t1=+) prob(t3=-|t2=+|t1=+)

            R114 = ramsey(angle*2,w1,w2,-np.exp(4*pi/3*1j)*l,-np.exp(4*pi/3*1j)*l,d,d,R112_prob[3],r1t31,r1t32,t,3,3)
            R114_prob = measure(R114) #  prob(t3=+|t2=-|t1=+) prob(t3=-|t2=-|t1=+)

            R115 = ramsey(angle*2,w1,w2,-np.exp(2*pi/3*1j)*l,-l,d,d,R111_prob[3],r1t21,r1t22,t,2,1)
           # R115 = FreeEvolution(R115,r1t21 + t, r1t31)
            R115_prob = measure(R115)  #prob(t2=+|t1=-) prob(t2=-|t1=-)

            R116 = ramsey(angle*2,w1,w2,-np.exp(4*pi/3*1j)*l,-np.exp(4*pi/3*1j)*l,d,d, notgate*R115_prob[2],r1t31,r1t32,t,3,3)
            R116_prob = measure(R116) # prob(t3=+|t2=+|t1=-) prob(t3=-|t2=+|t1=-)

            R117 = ramsey(angle*2,w1,w2,-np.exp(4*pi/3*1j)*l,-np.exp(4*pi/3*1j)*l,d,d,R115_prob[3],r1t31,r1t32,t,3,3)
            R117_prob = measure(R117) # prob(t3=+|t2=-|t1=-) prob(t3=-|t2=-|t1=-)


            # Now we find the expectation value of the first term in the expresion (A11I A12I A12R) in Eq. 10 of the paper 10.1103/PhysRevLett.114.250403


            R121 = ramsey(angle*2+pi/2,w1,w2,-l,-np.exp(2*pi/3*1j)*l,d,d,initial_state,r1t11,r1t12,t,1,2)
            #R121 = FreeEvolution(R121,r1t11 + t, r1t21) 
            R121_prob = measure(R121) # +,- prob(t1 =+),Prob(t1=-)


            R122 = ramsey(angle*2+pi/2,w1,w2,-np.exp(2*pi/3*1j)*l,-l,d,d,notgate*R121_prob[2],r1t21,r1t22,t,2,1)
            #R122 = FreeEvolution(R122,r1t21 + t, r1t31)
            R122_prob = measure(R122) # prob(t2=+|t1=+) prob(t2=-|t1=+)

            R123 = ramsey(angle*2,w1,w2,-np.exp(4*pi/3*1j)*l,-np.exp(4*pi/3*1j)*l,d,d,notgate*R122_prob[2],r1t31,r1t32,t,3,3)
            R123_prob = measure(R123) # prob(t3=+|t2=+|t1=+) prob(t3=-|t2=+|t1=+)

            R124 = ramsey(angle*2,w1,w2,-np.exp(4*pi/3*1j)*l,-np.exp(4*pi/3*1j)*l,d,d,R122_prob[3],r1t31,r1t32,t,3,3)
            R124_prob = measure(R124) #  prob(t3=+|t2=-|t1=+) prob(t3=-|t2=-|t1=+)

            R125 = ramsey(angle*2+pi/2,w1,w2,-np.exp(2*pi/3*1j)*l,-l,d,d,R121_prob[3],r1t21,r1t22,t,2,1)
           # R125 = FreeEvolution(R125,r1t21 + t, r1t31)
            R125_prob = measure(R125)  #prob(t2=+|t1=-) prob(t2=-|t1=-)

            R126 = ramsey(angle*2,w1,w2,-np.exp(4*pi/3*1j)*l,-np.exp(4*pi/3*1j)*l,d,d, notgate*R125_prob[2],r1t31,r1t32,t,3,3)
            R126_prob = measure(R126) # prob(t3=+|t2=+|t1=-) prob(t3=-|t2=+|t1=-)

            R127 = ramsey(angle*2,w1,w2,-np.exp(4*pi/3*1j)*l,-np.exp(4*pi/3*1j)*l,d,d,R125_prob[3],r1t31,r1t32,t,3,3)
            R127_prob = measure(R127) # prob(t3=+|t2=-|t1=-) prob(t3=-|t2=-|t1=-)

            # Now we find the expectation value of the first term in the expresion (A11I A12R A12I) in Eq. 10 of the paper 10.1103/PhysRevLett.114.250403


            R131 = ramsey(angle*2+pi/2,w1,w2,-l,-np.exp(2*pi/3*1j)*l,d,d,initial_state,r1t11,r1t12,t,1,2)
           # R131 = FreeEvolution(R131,r1t11 + t, r1t21)
            R131_prob = measure(R131) # +,- prob(t1 =+),Prob(t1=-)


            R132 = ramsey(angle*2,w1,w2,-np.exp(2*pi/3*1j)*l,-l,d,d,notgate*R131_prob[2],r1t21,r1t22,t,2,1)
            #R132 = FreeEvolution(R132,r1t21 + t, r1t31)
            R132_prob = measure(R132) # prob(t2=+|t1=+) prob(t2=-|t1=+)

            R133 = ramsey(angle*2+pi/2,w1,w2,-np.exp(4*pi/3*1j)*l,-np.exp(4*pi/3*1j)*l,d,d,notgate*R132_prob[2],r1t31,r1t32,t,3,3)
            R133_prob = measure(R133) # prob(t3=+|t2=+|t1=+) prob(t3=-|t2=+|t1=+)

            R134 = ramsey(angle*2+pi/2,w1,w2,-np.exp(4*pi/3*1j)*l,-np.exp(4*pi/3*1j)*l,d,d,R132_prob[3],r1t31,r1t32,t,3,3)
            R134_prob = measure(R134) #  prob(t3=+|t2=-|t1=+) prob(t3=-|t2=-|t1=+)

            R135 = ramsey(angle*2,w1,w2,-np.exp(2*pi/3*1j)*l,-l,d,d,R131_prob[3],r1t21,r1t22,t,2,1)
           # R135 = FreeEvolution(R135,r1t21 + t, r1t31)
            R135_prob = measure(R135)  #prob(t2=+|t1=-) prob(t2=-|t1=-)

            R136 = ramsey(angle*2+pi/2,w1,w2,-np.exp(4*pi/3*1j)*l,-np.exp(4*pi/3*1j)*l,d,d, notgate*R135_prob[2],r1t31,r1t32,t,3,3)
            R136_prob = measure(R136) # prob(t3=+|t2=+|t1=-) prob(t3=-|t2=+|t1=-)

            R137 = ramsey(angle*2+pi/2,w1,w2,-np.exp(4*pi/3*1j)*l,-np.exp(4*pi/3*1j)*l,d,d,R135_prob[3],r1t31,r1t32,t,3,3)
            R137_prob = measure(R137) # prob(t3=+|t2=-|t1=-) prob(t3=-|t2=-|t1=-)


            # Now we find the expectation value of the first term in the expresion (A11R A12I A12I) in Eq. 10 of the paper 10.1103/PhysRevLett.114.250403

            R141 = ramsey(angle*2,w1,w2,-l,-np.exp(2*pi/3*1j)*l,d,d,initial_state,r1t11,r1t12,t,1,2)
            #R141 = FreeEvolution(R141,r1t11 + t, r1t21)
            R141_prob = measure(R141) # +,- prob(t1 =+),Prob(t1=-)


            R142 = ramsey(angle*2+pi/2,w1,w2,-np.exp(2*pi/3*1j)*l,-l,d,d,notgate*R141_prob[2],r1t21,r1t22,t,2,1)
            #R142 = FreeEvolution(R142,r1t21 + t, r1t31)
            R142_prob = measure(R142) # prob(t2=+|t1=+) prob(t2=-|t1=+)

            R143 = ramsey(angle*2+pi/2,w1,w2,-np.exp(4*pi/3*1j)*l,-np.exp(4*pi/3*1j)*l,d,d,notgate*R142_prob[2],r1t31,r1t32,t,3,3)
            R143_prob = measure(R143) # prob(t3=+|t2=+|t1=+) prob(t3=-|t2=+|t1=+)

            R144 = ramsey(angle*2+pi/2,w1,w2,-np.exp(4*pi/3*1j)*l,-np.exp(4*pi/3*1j)*l,d,d,R142_prob[3],r1t31,r1t32,t,3,3)
            R144_prob = measure(R144) #  prob(t3=+|t2=-|t1=+) prob(t3=-|t2=-|t1=+)

            R145 = ramsey(angle*2+pi/2,w1,w2,-np.exp(2*pi/3*1j)*l,-l,d,d,R141_prob[3],r1t21,r1t22,t,2,1)
            #R145 = FreeEvolution(R145,r1t21 + t, r1t31)
            R145_prob = measure(R145)  #prob(t2=+|t1=-) prob(t2=-|t1=-)

            R146 = ramsey(angle*2+pi/2,w1,w2,-np.exp(4*pi/3*1j)*l,-np.exp(4*pi/3*1j)*l,d,d, notgate*R145_prob[2],r1t31,r1t32,t,3,3)
            R146_prob = measure(R146) # prob(t3=+|t2=+|t1=-) prob(t3=-|t2=+|t1=-)

            R147 = ramsey(angle*2+pi/2,w1,w2,-np.exp(4*pi/3*1j)*l,-np.exp(4*pi/3*1j)*l,d,d,R145_prob[3],r1t31,r1t32,t,3,3)
            R147_prob = measure(R147) # prob(t3=+|t2=-|t1=-) prob(t3=-|t2=-|t1=-)

            #three time corellation
            expectterm1= (R111_prob[0]*R112_prob[0]*(R113_prob[0]-R113_prob[1])) + (R111_prob[0]*R112_prob[1]*(-R114_prob[0]+R114_prob[1])) + (R111_prob[1]*R115_prob[0]*(-R116_prob[0]+R116_prob[1]))+(R111_prob[1]*R115_prob[1]*(R117_prob[0]-R117_prob[1]))
            expectterm2=(R121_prob[0]*R122_prob[0]*(R123_prob[0]-R123_prob[1])) + (R121_prob[0]*R122_prob[1]*(-R124_prob[0]+R124_prob[1])) + (R121_prob[1]*R125_prob[0]*(-R126_prob[0]+R126_prob[1]))+(R121_prob[1]*R125_prob[1]*(R127_prob[0]-R127_prob[1]))
            expectterm3=(R131_prob[0]*R132_prob[0]*(R133_prob[0]-R133_prob[1])) + (R131_prob[0]*R132_prob[1]*(-R134_prob[0]+R134_prob[1])) + (R131_prob[1]*R135_prob[0]*(-R136_prob[0]+R136_prob[1]))+(R131_prob[1]*R135_prob[1]*(R137_prob[0]-R137_prob[1]))
            expectterm4=(R141_prob[0]*R142_prob[0]*(R143_prob[0]-R143_prob[1])) + (R141_prob[0]*R142_prob[1]*(-R144_prob[0]+R144_prob[1])) + (R141_prob[1]*R145_prob[0]*(-R146_prob[0]+R146_prob[1]))+(R141_prob[1]*R145_prob[1]*(R147_prob[0]-R147_prob[1]))

            total=  (expectterm1 - expectterm2 - expectterm3 - expectterm4)
            
            return total


        R3.append(row3(r3t1,r3t1,r3t2,r3t2,r3t3,r3t3))


        def column1(r1t11,r1t12,r1t21,r1t22,r1t31,r1t32):
            # Now we find the expectation value of the first term in the expresion (A11R A12R A12R) in Eq. 10 of the paper 10.1103/PhysRevLett.114.250403
            #print(r1t11,r1t12,r1t21,r1t22,r1t31,r1t32)
            angle = (l**2/d**2)*(d*t - np.sin(d*t))/2


            R111 = ramsey(angle,w1,w2,l,0,d,d,initial_state,r1t11,r1t12,t,-1,0)
            #R111 = FreeEvolution(R111,r1t11 + t, r1t21)
            R111_prob = measure(R111) # +,- prob(t1 =+),Prob(t1=-)


            R112 = ramsey(angle,w1,w2,0,np.exp(2*pi/3*1j)*l,d,d,notgate*R111_prob[2],r1t21,r1t22,t,0,-2)
            #R112 = FreeEvolution(R112,r1t21 + t, r1t31)
            R112_prob = measure(R112) # prob(t2=+|t1=+) prob(t2=-|t1=+)

            R113 = ramsey(angle*2,w1,w2,-l,-np.exp(2*pi/3*1j)*l,d,d,notgate*R112_prob[2],r1t31,r1t32,t,1,2)
            R113_prob = measure(R113) # prob(t3=+|t2=+|t1=+) prob(t3=-|t2=+|t1=+)

            R114 = ramsey(angle*2,w1,w2,-l,-np.exp(2*pi/3*1j)*l,d,d,R112_prob[3],r1t31,r1t32,t,1,2)
            R114_prob = measure(R114) #  prob(t3=+|t2=-|t1=+) prob(t3=-|t2=-|t1=+)

            R115 = ramsey(angle,w1,w2,0,np.exp(2*pi/3*1j)*l,d,d,R111_prob[3],r1t21,r1t22,t,0,-2)
            #R115 =  FreeEvolution(R115,r1t21 + t, r1t31)
            R115_prob = measure(R115)  #prob(t2=+|t1=-) prob(t2=-|t1=-)

            R116 = ramsey(angle*2,w1,w2,-l,-np.exp(2*pi/3*1j)*l,d,d, notgate*R115_prob[2],r1t31,r1t32,t,1,2)
            R116_prob = measure(R116) # prob(t3=+|t2=+|t1=-) prob(t3=-|t2=+|t1=-)

            R117 = ramsey(angle*2,w1,w2,-l,np.exp(2*pi/3*1j)*l,d,d,R115_prob[3],r1t31,r1t32,t,1,2)
            R117_prob = measure(R117) # prob(t3=+|t2=-|t1=-) prob(t3=-|t2=-|t1=-)


            # Now we find the expectation value of the first term in the expresion (A11I A12I A12R) in Eq. 10 of the paper 10.1103/PhysRevLett.114.250403




            R121 = ramsey(angle+pi/2,w1,w2,l,0,d,d,initial_state,r1t11,r1t12,t,-1,0)
            R121_prob = measure(R121) # +,- prob(t1 =+),Prob(t1=-)

        
            R122 = ramsey(angle+pi/2,w1,w2,0,np.exp(2*pi/3*1j)*l,d,d,notgate*R121_prob[2],r1t21,r1t22,t,0,-2)
            R122_prob = measure(R122) # prob(t2=+|t1=+) prob(t2=-|t1=+)

            R123 = ramsey(angle*2,w1,w2,-l,-np.exp(2*pi/3*1j)*l,d,d,notgate*R122_prob[2],r1t31,r1t32,t,1,2)
            R123_prob = measure(R123) # prob(t3=+|t2=+|t1=+) prob(t3=-|t2=+|t1=+)

            R124 = ramsey(angle*2,w1,w2,-l,-np.exp(2*pi/3*1j)*l,d,d,R122_prob[3],r1t31,r1t32,t,1,2)
            R124_prob = measure(R124) #  prob(t3=+|t2=-|t1=+) prob(t3=-|t2=-|t1=+)

            R125 = ramsey(angle+pi/2,w1,w2,0,np.exp(2*pi/3*1j)*l,d,d,R121_prob[3],r1t21,r1t22,t,0,-2)
            R125_prob = measure(R125)  #prob(t2=+|t1=-) prob(t2=-|t1=-)

            R126 = ramsey(angle*2,w1,w2,-l,-np.exp(2*pi/3*1j)*l,d,d, notgate*R125_prob[2],r1t31,r1t32,t,1,2)
            R126_prob = measure(R126) # prob(t3=+|t2=+|t1=-) prob(t3=-|t2=+|t1=-)

            R127 = ramsey(angle*2,w1,w2,-l,-np.exp(2*pi/3*1j)*l,d,d,R125_prob[3],r1t31,r1t32,t,1,2)
            R127_prob = measure(R127) # prob(t3=+|t2=-|t1=-) prob(t3=-|t2=-|t1=-)

            # Now we find the expectation value of the first term in the expresion (A11I A12R A12I) in Eq. 10 of the paper 10.1103/PhysRevLett.114.250403



            R131 = ramsey(angle+pi/2,w1,w2,l,0,d,d,initial_state,r1t11,r1t12,t,-1,0)
            R131_prob = measure(R131) # +,- prob(t1 =+),Prob(t1=-)


            R132 = ramsey(angle,w1,w2,0,np.exp(2*pi/3*1j)*l,d,d,notgate*R131_prob[2],r1t21,r1t22,t,0,-2)
            R132_prob = measure(R132) # prob(t2=+|t1=+) prob(t2=-|t1=+)

            R133 = ramsey(angle*2+pi/2,w1,w2,-l,-np.exp(2*pi/3*1j)*l,d,d,notgate*R132_prob[2],r1t31,r1t32,t,1,2)
            R133_prob = measure(R133) # prob(t3=+|t2=+|t1=+) prob(t3=-|t2=+|t1=+)

            R134 = ramsey(angle*2+pi/2,w1,w2,-l,-np.exp(2*pi/3*1j)*l,d,d,R132_prob[3],r1t31,r1t32,t,1,2)
            R134_prob = measure(R134) #  prob(t3=+|t2=-|t1=+) prob(t3=-|t2=-|t1=+)

            R135 = ramsey(angle,w1,w2,0,np.exp(2*pi/3*1j)*l,d,d,R131_prob[3],r1t21,r1t22,t,0,-2)
            R135_prob = measure(R135)  #prob(t2=+|t1=-) prob(t2=-|t1=-)

            R136 = ramsey(angle*2+pi/2,w1,w2,-l,-np.exp(2*pi/3*1j)*l,d,d, notgate*R135_prob[2],r1t31,r1t32,t,1,2)
            R136_prob = measure(R136) # prob(t3=+|t2=+|t1=-) prob(t3=-|t2=+|t1=-)

            R137 = ramsey(angle*2+pi/2,w1,w2,-l,-np.exp(2*pi/3*1j)*l,d,d,R135_prob[3],r1t31,r1t32,t,1,2)
            R137_prob = measure(R137) # prob(t3=+|t2=-|t1=-) prob(t3=-|t2=-|t1=-)


            # Now we find the expectation value of the first term in the expresion (A11R A12I A12I) in Eq. 10 of the paper 10.1103/PhysRevLett.114.250403

            R141 = ramsey(angle,w1,w2,l,0,d,d,initial_state,r1t11,r1t12,t,-1,0)
            R141_prob = measure(R141) # +,- prob(t1 =+),Prob(t1=-)


            R142 = ramsey(angle+pi/2,w1,w2,0,np.exp(2*pi/3*1j)*l,d,d,notgate*R141_prob[2],r1t21,r1t22,t,0,-2)
            R142_prob = measure(R142) # prob(t2=+|t1=+) prob(t2=-|t1=+)

            R143 = ramsey(angle*2+pi/2,w1,w2,-l,-np.exp(2*pi/3*1j)*l,d,d,notgate*R142_prob[2],r1t31,r1t32,t,1,2)
            R143_prob = measure(R143) # prob(t3=+|t2=+|t1=+) prob(t3=-|t2=+|t1=+)

            R144 = ramsey(angle*2+pi/2,w1,w2,-l,-np.exp(2*pi/3*1j)*l,d,d,R142_prob[3],r1t31,r1t32,t,1,2)
            R144_prob = measure(R144) #  prob(t3=+|t2=-|t1=+) prob(t3=-|t2=-|t1=+)

            R145 = ramsey(angle+pi/2,w1,w2,0,np.exp(2*pi/3*1j)*l,d,d,R141_prob[3],r1t21,r1t22,t,0,-2)
            R145_prob = measure(R145)  #prob(t2=+|t1=-) prob(t2=-|t1=-)

            R146 = ramsey(angle*2+pi/2,w1,w2,-l,-np.exp(2*pi/3*1j)*l,d,d, notgate*R145_prob[2],r1t31,r1t32,t,1,2)
            R146_prob = measure(R146) # prob(t3=+|t2=+|t1=-) prob(t3=-|t2=+|t1=-)

            R147 = ramsey(angle*2+pi/2,w1,w2,-l,-np.exp(2*pi/3*1j)*l,d,d,R145_prob[3],r1t31,r1t32,t,1,2)
            R147_prob = measure(R147) # prob(t3=+|t2=-|t1=-) prob(t3=-|t2=-|t1=-)

            #three time corellation
            expectterm1= (R111_prob[0]*R112_prob[0]*(R113_prob[0]-R113_prob[1])) + (R111_prob[0]*R112_prob[1]*(-R114_prob[0]+R114_prob[1])) + (R111_prob[1]*R115_prob[0]*(-R116_prob[0]+R116_prob[1]))+(R111_prob[1]*R115_prob[1]*(R117_prob[0]-R117_prob[1]))
            expectterm2=(R121_prob[0]*R122_prob[0]*(R123_prob[0]-R123_prob[1])) + (R121_prob[0]*R122_prob[1]*(-R124_prob[0]+R124_prob[1])) + (R121_prob[1]*R125_prob[0]*(-R126_prob[0]+R126_prob[1]))+(R121_prob[1]*R125_prob[1]*(R127_prob[0]-R127_prob[1]))
            expectterm3=(R131_prob[0]*R132_prob[0]*(R133_prob[0]-R133_prob[1])) + (R131_prob[0]*R132_prob[1]*(-R134_prob[0]+R134_prob[1])) + (R131_prob[1]*R135_prob[0]*(-R136_prob[0]+R136_prob[1]))+(R131_prob[1]*R135_prob[1]*(R137_prob[0]-R137_prob[1]))
            expectterm4=(R141_prob[0]*R142_prob[0]*(R143_prob[0]-R143_prob[1])) + (R141_prob[0]*R142_prob[1]*(-R144_prob[0]+R144_prob[1])) + (R141_prob[1]*R145_prob[0]*(-R146_prob[0]+R146_prob[1]))+(R141_prob[1]*R145_prob[1]*(R147_prob[0]-R147_prob[1]))

            total=  (expectterm1 - expectterm2 - expectterm3 - expectterm4)
            
            return total

        C1.append(column1(c1t1,c1t1,c1t2,c1t2,c1t3,c1t3))

        def column2(r1t11,r1t12,r1t21,r1t22,r1t31,r1t32):
            # Now we find the expectation value of the first term in the expresion (A11R A12R A12R) in Eq. 10 of the paper 10.1103/PhysRevLett.114.250403
            #print(r1t11,r1t12,r1t21,r1t22,r1t31,r1t32)
            angle = (l**2/d**2)*(d*t - np.sin(d*t))/2


            R111 = ramsey(angle,w1,w2,0,l,d,d,initial_state,r1t11,r1t12,t,0,-1)
            #R111 = FreeEvolution(R111,r1t11 + t, r1t21)
            R111_prob = measure(R111) # +,- prob(t1 =+),Prob(t1=-)


            R112 = ramsey(angle,w1,w2,np.exp(2*pi/3*1j)*l,0,d,d,notgate*R111_prob[2],r1t21,r1t22,t,-2,0)
            #R112 = FreeEvolution(R112,r1t21 + t, r1t31)
            R112_prob = measure(R112) # prob(t2=+|t1=+) prob(t2=-|t1=+)

            R113 = ramsey(angle*2,w1,w2,-np.exp(2*pi/3*1j)*l,-l,d,d,notgate*R112_prob[2],r1t31,r1t32,t,2,1)
            R113_prob = measure(R113) # prob(t3=+|t2=+|t1=+) prob(t3=-|t2=+|t1=+)

            R114 = ramsey(angle*2,w1,w2,-np.exp(2*pi/3*1j)*l,-l,d,d,R112_prob[3],r1t31,r1t32,t,2,1)
            R114_prob = measure(R114) #  prob(t3=+|t2=-|t1=+) prob(t3=-|t2=-|t1=+)

            R115 = ramsey(angle,w1,w2,np.exp(2*pi/3*1j)*l,0,d,d,R111_prob[3],r1t21,r1t22,t,-2,0)
            #R115 =  FreeEvolution(R115,r1t21 + t, r1t31)
            R115_prob = measure(R115)  #prob(t2=+|t1=-) prob(t2=-|t1=-)

            R116 = ramsey(angle*2,w1,w2,-np.exp(2*pi/3*1j)*l,-l,d,d, notgate*R115_prob[2],r1t31,r1t32,t,2,1)
            R116_prob = measure(R116) # prob(t3=+|t2=+|t1=-) prob(t3=-|t2=+|t1=-)

            R117 = ramsey(angle*2,w1,w2,-np.exp(2*pi/3*1j)*l,-l,d,d,R115_prob[3],r1t31,r1t32,t,2,1)
            R117_prob = measure(R117) # prob(t3=+|t2=-|t1=-) prob(t3=-|t2=-|t1=-)


            # Now we find the expectation value of the first term in the expresion (A11I A12I A12R) in Eq. 10 of the paper 10.1103/PhysRevLett.114.250403



            R121 = ramsey(angle+pi/2,w1,w2,0,l,d,d,initial_state,r1t11,r1t12,t,0,-1)
            R121_prob = measure(R121) # +,- prob(t1 =+),Prob(t1=-)


            R122 = ramsey(angle+pi/2,w1,w2,np.exp(2*pi/3*1j)*l,0,d,d,notgate*R121_prob[2],r1t21,r1t22,t,-2,0)
            R122_prob = measure(R122) # prob(t2=+|t1=+) prob(t2=-|t1=+)

            R123 = ramsey(angle*2,w1,w2,-np.exp(2*pi/3*1j)*l,-l,d,d,notgate*R122_prob[2],r1t31,r1t32,t,2,1)
            R123_prob = measure(R123) # prob(t3=+|t2=+|t1=+) prob(t3=-|t2=+|t1=+)

            R124 = ramsey(angle*2,w1,w2,-np.exp(2*pi/3*1j)*l,-l,d,d,R122_prob[3],r1t31,r1t32,t,2,1)
            R124_prob = measure(R124) #  prob(t3=+|t2=-|t1=+) prob(t3=-|t2=-|t1=+)

            R125 = ramsey(angle+pi/2,w1,w2,np.exp(2*pi/3*1j)*l,0,d,d,R121_prob[3],r1t21,r1t22,t,-2,0)
            R125_prob = measure(R125)  #prob(t2=+|t1=-) prob(t2=-|t1=-)

            R126 = ramsey(angle*2,w1,w2,-np.exp(2*pi/3*1j)*l,-l,d,d, notgate*R125_prob[2],r1t31,r1t32,t,2,1)
            R126_prob = measure(R126) # prob(t3=+|t2=+|t1=-) prob(t3=-|t2=+|t1=-)

            R127 = ramsey(angle*2,w1,w2,-np.exp(2*pi/3*1j)*l,-l,d,d,R125_prob[3],r1t31,r1t32,t,2,1)
            R127_prob = measure(R127) # prob(t3=+|t2=-|t1=-) prob(t3=-|t2=-|t1=-)

            # Now we find the expectation value of the first term in the expresion (A11I A12R A12I) in Eq. 10 of the paper 10.1103/PhysRevLett.114.250403



            R131 = ramsey(angle+pi/2,w1,w2,0,l,d,d,initial_state,r1t11,r1t12,t,0,-1)
            R131_prob = measure(R131) # +,- prob(t1 =+),Prob(t1=-)


            R132 = ramsey(angle,w1,w2,np.exp(2*pi/3*1j)*l,0,d,d,notgate*R131_prob[2],r1t21,r1t22,t,-2,0)
            R132_prob = measure(R132) # prob(t2=+|t1=+) prob(t2=-|t1=+)

            R133 = ramsey(angle*2+pi/2,w1,w2,-np.exp(2*pi/3*1j)*l,-l,d,d,notgate*R132_prob[2],r1t31,r1t32,t,2,1)
            R133_prob = measure(R133) # prob(t3=+|t2=+|t1=+) prob(t3=-|t2=+|t1=+)

            R134 = ramsey(angle*2+pi/2,w1,w2,-np.exp(2*pi/3*1j)*l,-l,d,d,R132_prob[3],r1t31,r1t32,t,2,1)
            R134_prob = measure(R134) #  prob(t3=+|t2=-|t1=+) prob(t3=-|t2=-|t1=+)

            R135 = ramsey(angle,w1,w2,np.exp(2*pi/3*1j)*l,0,d,d,R131_prob[3],r1t21,r1t22,t,-2,0)
            R135_prob = measure(R135)  #prob(t2=+|t1=-) prob(t2=-|t1=-)

            R136 = ramsey(angle*2+pi/2,w1,w2,-np.exp(2*pi/3*1j)*l,-l,d,d, notgate*R135_prob[2],r1t31,r1t32,t,2,1)
            R136_prob = measure(R136) # prob(t3=+|t2=+|t1=-) prob(t3=-|t2=+|t1=-)

            R137 = ramsey(angle*2+pi/2,w1,w2,-np.exp(2*pi/3*1j)*l,-l,d,d,R135_prob[3],r1t31,r1t32,t,2,1)
            R137_prob = measure(R137) # prob(t3=+|t2=-|t1=-) prob(t3=-|t2=-|t1=-)


            # Now we find the expectation value of the first term in the expresion (A11R A12I A12I) in Eq. 10 of the paper 10.1103/PhysRevLett.114.250403

            R141 = ramsey(angle,w1,w2,0,l,d,d,initial_state,r1t11,r1t12,t,0,-1)
            R141_prob = measure(R141) # +,- prob(t1 =+),Prob(t1=-)


            R142 = ramsey(angle+pi/2,w1,w2,np.exp(2*pi/3*1j)*l,0,d,d,notgate*R141_prob[2],r1t21,r1t22,t,-2,0)
            R142_prob = measure(R142) # prob(t2=+|t1=+) prob(t2=-|t1=+)

            R143 = ramsey(angle*2+pi/2,w1,w2,-np.exp(2*pi/3*1j)*l,-l,d,d,notgate*R142_prob[2],r1t31,r1t32,t,2,1)
            R143_prob = measure(R143) # prob(t3=+|t2=+|t1=+) prob(t3=-|t2=+|t1=+)

            R144 = ramsey(angle*2+pi/2,w1,w2,-np.exp(2*pi/3*1j)*l,-l,d,d,R142_prob[3],r1t31,r1t32,t,2,1)
            R144_prob = measure(R144) #  prob(t3=+|t2=-|t1=+) prob(t3=-|t2=-|t1=+)

            R145 = ramsey(angle+pi/2,w1,w2,np.exp(2*pi/3*1j)*l,0,d,d,R141_prob[3],r1t21,r1t22,t,-2,0)
            R145_prob = measure(R145)  #prob(t2=+|t1=-) prob(t2=-|t1=-)

            R146 = ramsey(angle*2+pi/2,w1,w2,-np.exp(2*pi/3*1j)*l,-l,d,d, notgate*R145_prob[2],r1t31,r1t32,t,2,1)
            R146_prob = measure(R146) # prob(t3=+|t2=+|t1=-) prob(t3=-|t2=+|t1=-)

            R147 = ramsey(angle*2+pi/2,w1,w2,-np.exp(2*pi/3*1j)*l,-l,d,d,R145_prob[3],r1t31,r1t32,t,2,1)
            R147_prob = measure(R147) # prob(t3=+|t2=-|t1=-) prob(t3=-|t2=-|t1=-)

            #three time corellation
            expectterm1= (R111_prob[0]*R112_prob[0]*(R113_prob[0]-R113_prob[1])) + (R111_prob[0]*R112_prob[1]*(-R114_prob[0]+R114_prob[1])) + (R111_prob[1]*R115_prob[0]*(-R116_prob[0]+R116_prob[1]))+(R111_prob[1]*R115_prob[1]*(R117_prob[0]-R117_prob[1]))
            expectterm2=(R121_prob[0]*R122_prob[0]*(R123_prob[0]-R123_prob[1])) + (R121_prob[0]*R122_prob[1]*(-R124_prob[0]+R124_prob[1])) + (R121_prob[1]*R125_prob[0]*(-R126_prob[0]+R126_prob[1]))+(R121_prob[1]*R125_prob[1]*(R127_prob[0]-R127_prob[1]))
            expectterm3=(R131_prob[0]*R132_prob[0]*(R133_prob[0]-R133_prob[1])) + (R131_prob[0]*R132_prob[1]*(-R134_prob[0]+R134_prob[1])) + (R131_prob[1]*R135_prob[0]*(-R136_prob[0]+R136_prob[1]))+(R131_prob[1]*R135_prob[1]*(R137_prob[0]-R137_prob[1]))
            expectterm4=(R141_prob[0]*R142_prob[0]*(R143_prob[0]-R143_prob[1])) + (R141_prob[0]*R142_prob[1]*(-R144_prob[0]+R144_prob[1])) + (R141_prob[1]*R145_prob[0]*(-R146_prob[0]+R146_prob[1]))+(R141_prob[1]*R145_prob[1]*(R147_prob[0]-R147_prob[1]))

            total=  (expectterm1 - expectterm2 - expectterm3 - expectterm4)
            
            return total

        C2.append(column2(c2t1,c2t1,c2t2,c2t2,c2t3,c2t3))



        def column3(r1t11,r1t12,r1t21,r1t22,r1t31,r1t32):
            # Now we find the expectation value of the first term in the expresion (A11R A12R A12R) in Eq. 10 of the paper 10.1103/PhysRevLett.114.250403
            #print(r1t11,r1t12,r1t21,r1t22,r1t31,r1t32)
            angle = (l**2/d**2)*(d*t - np.sin(d*t))/2


            R111 = ramsey(angle*2,w1,w2,l,l,d,d,initial_state,r1t11,r1t12,t,1,1)
            #R111 = FreeEvolution(R111,r1t11 + t, r1t21)
            R111_prob = measure(R111) # +,- prob(t1 =+),Prob(t1=-)


            R112 = ramsey(angle*2,w1,w2,l,l,d,d,notgate*R111_prob[2],r1t21,r1t22,t,2,2)
            #R112 = FreeEvolution(R112,r1t21 + t, r1t31)
            R112_prob = measure(R112) # prob(t2=+|t1=+) prob(t2=-|t1=+)

            R113 = ramsey(angle*2,w1,w2,l,l,d,d,notgate*R112_prob[2],r1t31,r1t32,t,3,3)
            R113_prob = measure(R113) # prob(t3=+|t2=+|t1=+) prob(t3=-|t2=+|t1=+)

            R114 = ramsey(angle*2,w1,w2,l,l,d,d,R112_prob[3],r1t31,r1t32,t,3,3)
            R114_prob = measure(R114) #  prob(t3=+|t2=-|t1=+) prob(t3=-|t2=-|t1=+)

            R115 = ramsey(angle*2,w1,w2,l,l,d,d,R111_prob[3],r1t21,r1t22,t,2,2)
            #R115 =  FreeEvolution(R115,r1t21 + t, r1t31)
            R115_prob = measure(R115)  #prob(t2=+|t1=-) prob(t2=-|t1=-)

            R116 = ramsey(angle*2,w1,w2,l,l,d,d, notgate*R115_prob[2],r1t31,r1t32,t,3,3)
            R116_prob = measure(R116) # prob(t3=+|t2=+|t1=-) prob(t3=-|t2=+|t1=-)

            R117 = ramsey(angle*2,w1,w2,l,l,d,d,R115_prob[3],r1t31,r1t32,t,3,3)
            R117_prob = measure(R117) # prob(t3=+|t2=-|t1=-) prob(t3=-|t2=-|t1=-)


            # Now we find the expectation value of the first term in the expresion (A11I A12I A12R) in Eq. 10 of the paper 10.1103/PhysRevLett.114.250403

            



            R121 = ramsey(angle*2+pi/2,w1,w2,l,l,d,d,initial_state,r1t11,r1t12,t,1,1)
            R121_prob = measure(R121) # +,- prob(t1 =+),Prob(t1=-)


            R122 = ramsey(angle*2+pi/2,w1,w2,l,l,d,d,notgate*R121_prob[2],r1t21,r1t22,t,2,2)
            R122_prob = measure(R122) # prob(t2=+|t1=+) prob(t2=-|t1=+)

            R123 = ramsey(angle*2,w1,w2,l,l,d,d,notgate*R122_prob[2],r1t31,r1t32,t,3,3)
            R123_prob = measure(R123) # prob(t3=+|t2=+|t1=+) prob(t3=-|t2=+|t1=+)

            R124 = ramsey(angle*2,w1,w2,l,l,d,d,R122_prob[3],r1t31,r1t32,t,3,3)
            R124_prob = measure(R124) #  prob(t3=+|t2=-|t1=+) prob(t3=-|t2=-|t1=+)

            R125 = ramsey(angle*2+pi/2,w1,w2,l,l,d,d,R121_prob[3],r1t21,r1t22,t,2,2)
            R125_prob = measure(R125)  #prob(t2=+|t1=-) prob(t2=-|t1=-)

            R126 = ramsey(angle*2,w1,w2,l,l,d,d, notgate*R125_prob[2],r1t31,r1t32,t,3,3)
            R126_prob = measure(R126) # prob(t3=+|t2=+|t1=-) prob(t3=-|t2=+|t1=-)

            R127 = ramsey(angle*2,w1,w2,l,l,d,d,R125_prob[3],r1t31,r1t32,t,3,3)
            R127_prob = measure(R127) # prob(t3=+|t2=-|t1=-) prob(t3=-|t2=-|t1=-)

            # Now we find the expectation value of the first term in the expresion (A11I A12R A12I) in Eq. 10 of the paper 10.1103/PhysRevLett.114.250403



            R131 = ramsey(angle*2+pi/2,w1,w2,l,l,d,d,initial_state,r1t11,r1t12,t,1,1)
            R131_prob = measure(R131) # +,- prob(t1 =+),Prob(t1=-)


            R132 = ramsey(angle*2,w1,w2,l,l,d,d,notgate*R131_prob[2],r1t21,r1t22,t,2,2)
            R132_prob = measure(R132) # prob(t2=+|t1=+) prob(t2=-|t1=+)

            R133 = ramsey(angle*2+pi/2,w1,w2,l,l,d,d,notgate*R132_prob[2],r1t31,r1t32,t,3,3)
            R133_prob = measure(R133) # prob(t3=+|t2=+|t1=+) prob(t3=-|t2=+|t1=+)

            R134 = ramsey(angle*2+pi/2,w1,w2,l,l,d,d,R132_prob[3],r1t31,r1t32,t,3,3)
            R134_prob = measure(R134) #  prob(t3=+|t2=-|t1=+) prob(t3=-|t2=-|t1=+)

            R135 = ramsey(angle*2,w1,w2,l,l,d,d,R131_prob[3],r1t21,r1t22,t,2,2)
            R135_prob = measure(R135)  #prob(t2=+|t1=-) prob(t2=-|t1=-)

            R136 = ramsey(angle*2+pi/2,w1,w2,l,l,d,d, notgate*R135_prob[2],r1t31,r1t32,t,3,3)
            R136_prob = measure(R136) # prob(t3=+|t2=+|t1=-) prob(t3=-|t2=+|t1=-)

            R137 = ramsey(angle*2+pi/2,w1,w2,l,l,d,d,R135_prob[3],r1t31,r1t32,t,3,3)
            R137_prob = measure(R137) # prob(t3=+|t2=-|t1=-) prob(t3=-|t2=-|t1=-)


            # Now we find the expectation value of the first term in the expresion (A11R A12I A12I) in Eq. 10 of the paper 10.1103/PhysRevLett.114.250403

            R141 = ramsey(angle*2,w1,w2,l,l,d,d,initial_state,r1t11,r1t12,t,1,1)
            R141_prob = measure(R141) # +,- prob(t1 =+),Prob(t1=-)


            R142 = ramsey(angle*2+pi/2,w1,w2,l,l,d,d,notgate*R141_prob[2],r1t21,r1t22,t,2,2)
            R142_prob = measure(R142) # prob(t2=+|t1=+) prob(t2=-|t1=+)

            R143 = ramsey(angle*2+pi/2,w1,w2,l,l,d,d,notgate*R142_prob[2],r1t31,r1t32,t,3,3)
            R143_prob = measure(R143) # prob(t3=+|t2=+|t1=+) prob(t3=-|t2=+|t1=+)

            R144 = ramsey(angle*2+pi/2,w1,w2,l,l,d,d,R142_prob[3],r1t31,r1t32,t,3,3)
            R144_prob = measure(R144) #  prob(t3=+|t2=-|t1=+) prob(t3=-|t2=-|t1=+)

            R145 = ramsey(angle*2+pi/2,w1,w2,l,l,d,d,R141_prob[3],r1t21,r1t22,t,2,2)
            R145_prob = measure(R145)  #prob(t2=+|t1=-) prob(t2=-|t1=-)

            R146 = ramsey(angle*2+pi/2,w1,w2,l,l,d,d, notgate*R145_prob[2],r1t31,r1t32,t,3,3)
            R146_prob = measure(R146) # prob(t3=+|t2=+|t1=-) prob(t3=-|t2=+|t1=-)

            R147 = ramsey(angle*2+pi/2,w1,w2,l,l,d,d,R145_prob[3],r1t31,r1t32,t,3,3)
            R147_prob = measure(R147) # prob(t3=+|t2=-|t1=-) prob(t3=-|t2=-|t1=-)

            #three time corellation
            expectterm1= (R111_prob[0]*R112_prob[0]*(R113_prob[0]-R113_prob[1])) + (R111_prob[0]*R112_prob[1]*(-R114_prob[0]+R114_prob[1])) + (R111_prob[1]*R115_prob[0]*(-R116_prob[0]+R116_prob[1]))+(R111_prob[1]*R115_prob[1]*(R117_prob[0]-R117_prob[1]))
            expectterm2=(R121_prob[0]*R122_prob[0]*(R123_prob[0]-R123_prob[1])) + (R121_prob[0]*R122_prob[1]*(-R124_prob[0]+R124_prob[1])) + (R121_prob[1]*R125_prob[0]*(-R126_prob[0]+R126_prob[1]))+(R121_prob[1]*R125_prob[1]*(R127_prob[0]-R127_prob[1]))
            expectterm3=(R131_prob[0]*R132_prob[0]*(R133_prob[0]-R133_prob[1])) + (R131_prob[0]*R132_prob[1]*(-R134_prob[0]+R134_prob[1])) + (R131_prob[1]*R135_prob[0]*(-R136_prob[0]+R136_prob[1]))+(R131_prob[1]*R135_prob[1]*(R137_prob[0]-R137_prob[1]))
            expectterm4=(R141_prob[0]*R142_prob[0]*(R143_prob[0]-R143_prob[1])) + (R141_prob[0]*R142_prob[1]*(-R144_prob[0]+R144_prob[1])) + (R141_prob[1]*R145_prob[0]*(-R146_prob[0]+R146_prob[1]))+(R141_prob[1]*R145_prob[1]*(R147_prob[0]-R147_prob[1]))

            total=  (expectterm1 - expectterm2 - expectterm3 - expectterm4)
            
            return total
        C3.append(column3(c3t1,c3t1,c3t2,c3t2,c3t3,c3t3))
    
        chi.append(R1[i]+R2[i]+R3[i]+C1[i]+C2[i]-C3[i])
print(chi)
        
with open("Vacuum_Correlated_15Hz_CorrectedF.csv", "a", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(R1)
    writer.writerow(R2)
    writer.writerow(R3)
    writer.writerow(C1)
    writer.writerow(C2)
    writer.writerow(C3)
    writer.writerow(chi)






