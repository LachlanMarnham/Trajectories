#####################	FUNCTIONS	#####################

def V(px, py, b, gamma, J):
	# b = 1.0 or -1.0 is the dispersion branch
	p = math.sqrt(px**2 + py**2)
	phi = math.atan2(py,px)
	cos = math.cos(3 * phi)
	sin = math.sin(3 * phi)
	vx = px / p * b - 2.0 * gamma * px + 2*b*J*px*cos + 3*b*J*py*sin
	vy = py / p * b - 2.0 * gamma * py - 3*b*J*px*sin + 2*b*J*py*cos
	return vx, vy

def Force(dx, dy, alpha, d):
    r = math.sqrt(dx**2 + dy**2 + d**2)
    Fx = alpha * dx / r**3 
    Fy = alpha * dy / r**3
    return Fx, Fy
    
def rhs(x):
    x1 = x[0]
    y1 = x[1]
    x2 = x[2]
    y2 = x[3]
    px1 = x[4]
    py1 = x[5]
    px2 = x[6]
    py2 = x[7]
    k = np.zeros((8))
    
    vx1, vy1 = V(px1, py1,  1.0, gamma, J)
    vx2, vy2 = V(px2, py2, -1.0, gamma, J)
    k[0] = vx1
    k[1] = vy1
    k[2] = vx2
    k[3] = vy2

    Fx, Fy = Force(x1 - x2, y1 - y2, alpha,d)
    k[4] = Fx
    k[5] = Fy
    k[6] = -Fx
    k[7] = -Fy

    return k

def do_rk4step(x, tau):
        k1 = tau * rhs(x)
        k2 = tau * rhs(x + k1 / 2.0)
        k3 = tau * rhs(x + k2 / 2.0) 
        k4 = tau * rhs(x + k3)
        return x + 1.0 / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

def xpnorm(x):
    dxnorm = x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2
    dxnorm = math.sqrt(dxnorm)
    dpnorm = x[4]**2 + x[5]**2 + x[6]**2 + x[7]**2
    dpnorm = math.sqrt(dpnorm)
    return dxnorm, dpnorm
    
def rk4step(x_1, y_1, x_2, y_2, px_1, py_1, px_2, py_2, tau):
    x  = np.array([x_1, y_1, x_2, y_2, px_1, py_1, px_2, py_2])
    xnew = do_rk4step(x, tau)
    dxnorm, dpnorm = xpnorm(xnew - x)
    dx0 = 0.03
    dp0 = 0.03
    n_steps = int(max(dxnorm / dx0, dpnorm / dp0, 1)) 
    if n_steps > 1: 
        #print "large step: split into ", n_steps 
        x0 = x
        tau_s = tau / n_steps
        for i_t in range(n_steps):
            x1 = do_rk4step(x0, tau_s)
            x0 = x1
        xnew = x1    
    return xnew[0], xnew[1], xnew[2], xnew[3], xnew[4], xnew[5], xnew[6], xnew[7]

def angMom(x, y, px, py):
	theta_r = math.atan2(y,x)
	theta_p = math.atan2(py,px)
	theta = theta_p - theta_r
	r = math.sqrt(x**2 + y**2)
	p = math.sqrt(px**2 + py**2)

	L = r * p * math.sin(theta)

	return L

def energy(x1,y1,x2,y2,px1,py1,px2,py2,alpha,gamma,d):
	p1 = math.sqrt(px1**2 + py1**2)
	p2 = math.sqrt(px2**2 + py2**2)
	dx = x1 - x2
	dy = y1 - y2
	r = math.sqrt(dx**2 + dy**2 + d**2)

	E = p1 - p2 - gamma * (p1**2 + p2**2) + alpha/r
	
	return E

def makeFilename(theta,frac_p,frac_K):
	fName = 'data-traj-theta=%d-fracp=%g-fracK=%g.npz' % (theta, frac_p, frac_K)
	return fName

def saveVectors(f,x_1,y_1,x_2,y_2,px_1,py_1,px_2,py_2):
	print "Saving..."
	np.savez(f,x_1=x_1,y_1=y_1,x_2=x_2,y_2=y_2,px_1=px_1,py_1=py_1,px_2=px_2,py_2=py_2)
	return 0

def loadData(fname):
	print "Loading data..."
	data = np.load(fname)

	x_1 = data['x_1']
	y_1 = data['y_1']
	x_2 = data['x_2']
	y_2 = data['y_2']
	px_1 = data['px_1']
	py_1 = data['py_1']
	px_2 = data['px_2']
	py_2 = data['py_2']

	return x_1, y_1, x_2, y_2, px_1, py_1, px_2, py_2

def makeFigName(theta,frac_p,frac_K):
	fName = 'theta=%d-fracp=%g-fracK=%g.png' % (theta, frac_p, frac_K)
	return fName


#####################	MAIN	#####################


if __name__ == '__main__':
	import math
	import numpy as np
	import matplotlib.pyplot as mpl
	import os

	singleLayer = False
	theta_d = (50,90,130,170,210,250,290)
	frac_p = (0.1,0.3,0.5,0.7,0.9)
	frac_K = (0.1,0.3,0.5,0.7,0.9)
	timeStep = 0.01
	maxTime = 5000
	nm = 1e-9

	if singleLayer == True:
		h = 0 * nm
	else:
		h = 1.3 * nm
	
	q_e = 1.60217657e-19
	v_f = 1e6
	pi = math.pi
	eps_0 = 8.85418782e-12
	eps_s = 3.5					#dielectric const. (H-BN)
	eV = q_e
	t = 2.8 * eV
	t_prime = 0.1 * t
	angs = 1e-10
	a = 1.42 * angs				#carbon-carbon distance
	hbar = 1.05457173e-34
	a_0 = 0.5 * angs			#Bohr radius for t'=0.1t
	p_0 = hbar / a_0			#natural momentum scale
	eps_g = eps_s + q_e**2/8.0/pi/eps_0/hbar/v_f
	d = h / a_0
	m = hbar**2 / 9.0 / a**2 / t_prime
	mu = 3.0 * a**2 * t / 8.0 / hbar**2

	if singleLayer == True:
		seperation = 1
	else:
		omega = math.sqrt(q_e**2/4.0/pi/eps_0/eps_g/m/h**3)
		seperation = math.sqrt(2.0*hbar/m/omega) / a_0
		
	alpha = q_e**2/4.0/pi/eps_0/eps_g/v_f/p_0/a_0
	gamma = 9.0*a**2*t_prime*p_0/4.0/v_f/hbar**2
	J = mu * p_0 / v_f
	t_0 = a_0 / v_f				#natural time scale

	print J
	
	#define arrays
	t = np.arange(0,maxTime+timeStep,timeStep)
	x_1 = np.zeros((len(t)))
	y_1 = np.zeros((len(t)))
	x_2 = np.zeros((len(t)))
	y_2 = np.zeros((len(t)))
	px_1 = np.zeros((len(t)))
	py_1 = np.zeros((len(t)))
	px_2 = np.zeros((len(t)))
	py_2 = np.zeros((len(t)))
	L = np.zeros((len(t)))
	E = np.zeros((len(t)))
	
	for m in range(len(theta_d)):
		theta = theta_d[m] * pi / 180
		cos = math.cos(theta)
		c_1 = 6 * gamma - 4 * gamma * cos
		c_2 = math.sqrt(5.0 - 4.0 * cos) - 1.0
		c_3 = math.sqrt(c_2**2 + 4*c_1*c_2*alpha/seperation)
		p_max = 1.0/2.0/c_1*c_3 - c_2/2.0/c_1
		

		for n in range(len(frac_p)):
			p_mag = frac_p[n] * p_max
			K_max = 2 * p_mag
			
			for j in range(len(frac_K)):
				K_mag = frac_K[j] * K_max
				K_x = K_mag
				K_y = 0

				#initial conditions
				x_1[0] = seperation
				y_1[0] = 0
				x_2[0] = 0
				y_2[0] = 0
				px_1[0] = p_mag * math.cos(theta)
				py_1[0] = p_mag * math.sin(theta)
				px_2[0] = K_x - px_1[0]
				py_2[0] = K_y - py_1[0]
				#L_1_0 = angMom(x_1[0],y_1[0],px_1[0],py_1[0])
				#L_2_0 = angMom(x_2[0],y_2[0],px_2[0],py_2[0])        
				#L[0] = L_1_0 + L_2_0
				#E[0] = energy(x_1[0],y_1[0],x_2[0],y_2[0],px_1[0],py_1[0],px_2[0],py_2[0],alpha,gamma,d)

				#generate the file name
				f = makeFilename(theta_d[m],frac_p[n],frac_K[j])

				if os.path.isfile(f):
					x_1,y_1,x_2,y_2,px_1,py_1,px_2,py_2 = loadData(f)
				else:
					print "calculating..."
					for i in range(0,len(t)-1):
							x1n, y1n, x2n, y2n, px1n, py1n, px2n, py2n = rk4step(x_1[i], y_1[i], x_2[i], y_2[i], px_1[i], py_1[i], px_2[i], py_2[i], timeStep)
							x_1[i + 1] = x1n
							y_1[i + 1] = y1n
							x_2[i + 1] = x2n
							y_2[i + 1] = y2n
							px_1[i + 1] = px1n
							py_1[i + 1] = py1n
							px_2[i + 1] = px2n
							py_2[i + 1] = py2n
							#L_1 = angMom(x1n,y1n,px1n,py1n)
							#L_2 = angMom(x2n,y2n,px2n,py2n)

							#L[i + 1] = L_1 + L_2
							#E[i + 1] = energy(x1n,y1n,x2n,y2n,px1n,py1n,px2n,py2n,alpha,gamma,d)
							#print L_1, L_2
							
					saveVectors(f,x_1,y_1,x_2,y_2,px_1,py_1,px_2,py_2)
				
				#plot graphs
				mpl.plot(x_1*a_0/angs,y_1*a_0/angs, lw = 3)
				mpl.plot(x_2*a_0/angs,y_2*a_0/angs, lw = 3)
				mpl.xlabel('x (angstroms)')
				mpl.ylabel('y (angstroms)')
				mpl.xlim(min(min(x_1*a_0/angs),min(x_2*a_0/angs)),max(max(x_1*a_0/angs),max(x_2*a_0/angs)))
				mpl.ylim(min(min(y_1*a_0/angs),min(y_2*a_0/angs)),max(max(y_1*a_0/angs),max(y_2*a_0/angs)))
				g = makeFigName(theta_d[m],frac_p[n],frac_K[j])
				print g
				#mpl.show()
				mpl.savefig(g)
				mpl.close()
				#mpl.show()
				#mpl.plot(t, L, lw = 3)
				#mpl.show()
				#mpl.plot(t, E, lw = 3)
				#mpl.show()
