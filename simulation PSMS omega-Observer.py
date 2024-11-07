import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# To Make Plots Look Like Latex

# plt.rcParams.update({
#   "text.usetex": True,
#   "font.family": "sans-serif",
#   "font.sans-serif": ["Helvetica"]
# })

enableSave = True

class Parameters:
    def __init__(self):   
        # The stator windings are actually unbalanced
        self.J      = 0.012  #[Kg*m^2]
        self.beta   = 0.0026 #[Nms/rad]
        self.R      = 0.225  #[Ohm]
        self.Phi    = 0.17   #[Wb]
        self.L      = 0.0038 #[h]
        
        self.i_max  = 15     #[A]
        self.Q      = np.diag([1/self.L, 1/self.L, 1/self.J]) 
        #
        self.R_x    = np.array([[self.R,0,0],
                                [0,self.R,0],
                                [0,0,self.beta]
                                ])
        #
        #System matrices
        self.G      = np.array([[1,0],
                                [0,1],
                                [0,0]
                               ])

        self.gamma  = 10 #+0*self.i_max**2*self.L**2/(4*self.beta)

        # Initial Conditions
        self.x0    = np.array([0, 0.001, 1])
        self.xhat0 = np.array([0, 0, 0])


        # Total Time
        self.tf = 0.5


#System under consideration
def f(x, u, params):
    #State labeling
    phi_d         = x[0]
    phi_q         = x[1]
    p             = x[2]

    # # saturation of fluxes
    phi_d = sat(phi_d,params.i_max*params.L)
    phi_q = sat(phi_q,params.i_max*params.L)


    J_x = np.array([
                    [       0,                   0,                  phi_q],
                    [       0,                   0,   -(phi_d+ params.Phi)],
                    [  -phi_q, (phi_d+ params.Phi),                      0]
                    ])
    #
    # SYSTEM DYNAMICS
    xdot = (J_x - params.R).dot(params.Q.dot(x)) + params.G.dot(u)
    return xdot 

#Obsever dynamics
def obs(xhat, u, y, params):
    #State labeling
    phi_d         = xhat[0]
    phi_q         = xhat[1]
    p             = xhat[2]

    # # saturation of fluxes
    phi_d = sat(phi_d,params.i_max*params.L)
    phi_q = sat(phi_q,params.i_max*params.L)

    J_x = np.array([
                    [       0,                   0,                  phi_q],
                    [       0,                   0,   -(phi_d+ params.Phi)],
                    [  -phi_q, (phi_d+ params.Phi),                      0]
                    ])
    #
    y_hat = params.G.T.dot(params.Q.dot(xhat))
    #Dynamics
    xhatdot = (J_x - params.R ).dot(params.Q.dot(xhat)) + params.G.dot(u) + params.gamma*params.G.dot(y-y_hat)

    # xhatdot = (J_x - params.R - params.gamma*params.G.dot(params.G.T)).dot(params.Q.dot(xhat)) + params.G.dot(u) + params.gamma*params.G.dot(y)
    return xhatdot 


#Saturation function
def sat(x,L):
    if (-L <= x <= L ):
        return x
    else:
        return L*np.sign(x)
    

# #State feedback Control law 
# def ControlLaw(x, params):
#     x_1        = x[0]
#     x_2        = x[1]
#     rho        = params.rho
#     #Control Law definition
#     xsat       = np.array([sat(x_1,rho),sat(x_2,rho)])
#     K          = params.K
#     return -K.dot(xsat)

#Closed Loop dynamics
def odefun(t, xCL, params):
    # Order of the system
    # n_x+n_xhat
    
    n_x     = 3
    n_xhat  = 3

    x       = xCL[0            : n_x ]
    xhat    = xCL[n_x          : n_x + n_xhat]


    #Controller definition
    u = np.array([-4*np.cos(10*t),3*np.cos(3*t)]) # ControlLaw(xhat, params)
    # u = np.array([-4,3]) # ControlLaw(xhat, params)

    #Output definition
    y = params.G.T.dot(params.Q.dot(x))

    #Dynamcis Definition
    xdot    = f(   x, u, params)
    xhatdot = obs(xhat, u, y, params)
    xCLdot  = np.concatenate([xdot,xhatdot])
    return xCLdot


# Load Parameters
params = Parameters()
# Initial condition
xcl_0 = np.concatenate([params.x0, params.xhat0])
#print(np.zeros((2,1)))

print(params.gamma)

# With observer stiffness problem, other methods are LSODA, BDF
sol = solve_ivp(lambda t, xcl: odefun(t, xcl, params), [0.0, params.tf], xcl_0, method='LSODA', t_eval=np.arange(0.0, params.tf, 0.001))

xCL         = sol.y
t           = sol.t

x           =  xCL[0 : 3 ]
xhat        =  xCL[3 : 3 + 3]
omega       =    x[2,:]
omega_hat   = xhat[2,:]


phi_d_tilde  = xhat[0,:] - x[0,:] 
phi_q_tilde  = xhat[1,:] - x[1,:]
omega_tilde  = omega_hat - omega

# print(phi_d_tilde[10**3*params.tf-1])
# print(phi_q_tilde[10**3*params.tf-1])
# print(omega_tilde[10**3*params.tf-1])


y_tilde      = params.G.T.dot(xhat-x)

fig = plt.figure()
plt.plot( t,     x[0,:],           color='red',     label = r'$\varphi_D(t) $')
plt.plot( t,     x[1,:],      color='darkblue',     label = r'$\varphi_Q(t) $')
plt.plot( t,     omega,          color='green', label = r'$\omega(t) $')
plt.plot( t,     xhat[0,:],     color='orange',     label = r'$ \hat \varphi_D(t) $')
plt.plot( t,     xhat[1,:],  color='lightblue',     label = r'$\hat \varphi_Q(t) $')
plt.plot( t,     omega_hat, color='lightgreen', label = r'$ \hat \omega(t) $')
plt.title(r'Real and estimated states')
plt.xlim((0,params.tf))
plt.grid()

if enableSave:
    import tikzplotlib
    tikzplotlib.save("/home/mario/Documents/LATEX/images/PMSM_states_u_cos.tex")
else:
    plt.legend(loc='upper right')


# fig = plt.figure()
# plt.plot( t,     -4*np.cos(10*t),  color='red', label = r'$v_d(t) $')
# plt.plot( t,       3*np.cos(3*t), color='blue', label = r'$v_q(t) $')
# plt.title(r'Input $u(t)$')
# plt.xlim((0,params.tf))
# plt.grid()

# if enableSave:
#     import tikzplotlib
#     tikzplotlib.save("/home/mario/Documents/LATEX/images/PMSM_u_cost.tex")
# else:
#     plt.legend(loc='upper right')


fig = plt.figure()
plt.plot( t,     phi_d_tilde, color='red', label = r'$\tilde \varphi_D(t) $')
plt.plot( t,     phi_q_tilde,  color='darkblue', label = r'$\tilde \varphi_Q(t) $')
plt.plot( t,     omega_tilde, color='green', label = r'$ \tilde \omega(t) $')
plt.title(r'Error evolution')
plt.xlim((0,params.tf))
plt.grid()

if enableSave:
    import tikzplotlib
    # tikzplotlib.save("/home/mario/Documents/LATEX/images/PMSM_error_states_u_cost.tex")
else:
    plt.legend(loc='upper right')

if enableSave==False:
    plt.show()