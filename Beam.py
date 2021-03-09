from dolfin import *
import math
from matplotlib import  pyplot as plt
import numpy as np

from sympy import symbols
from sympy import ccode
import sympy as sp

import test


class GN_bending:
	def __init__(self, E, G, mesh):
		self.Exxxx = E[0]
		self.Eyyyy = E[1]
		self.Ezzzz = E[2]
		self.Eyyzz = E[3]
		self.Exxzz = E[4]
		self.Exxyy = E[5]
		self.Gyz = G[0]
		self.Gxz = G[1]
		self.Gxy = G[2]
		self.mesh = mesh

def solver_GN(f, df, ddf, num_elems, bcs_case, F_ext_cor, F_ext):
    """
    Solve Timoshenko beam problem 
    
    f - distributed load function
    num_elems - number of elements
    
    """
    x_left = 0.0
    x_right = L_beam
    mesh = IntervalMesh(num_elems, x_left, x_right)
    
    P = FiniteElement("CG", interval, 1)
    P2 = FiniteElement("CG", interval, 2)
    P0 = FiniteElement("CG", interval, 2)
    
    element = MixedElement([P2, P])
    V = FunctionSpace(mesh, element)
    W = FunctionSpace(mesh, P0)
    W1 = FunctionSpace(mesh, P)
    u_test = TestFunction(V)
    w_t, fi_t  = split(u_test)
        
    u_trial = TrialFunction(V)
    w, fi = split(u_trial)
    
    #boundary conditions
    def both_ends(x, on_boundary):
        return on_boundary
    def left_end(x, on_boundary):
        return near(x[0], x_left) and on_boundary
    def right_end(x, on_boundary):
        return near(x[0], x_right) and on_boundary

    #bc = DirichletBC(V.sub(0), Constant(0.), both_ends)
    if bcs_case == 'C-H':
        bc = [DirichletBC(V.sub(0), Constant(0.), both_ends), DirichletBC(V.sub(1), Constant(0.), left_end)]
    if bcs_case == 'C-C':
        bc = [DirichletBC(V.sub(0), Constant(0.), both_ends), DirichletBC(V.sub(1), Constant(0.), both_ends)]
    if bcs_case == "H-H":
        bc = DirichletBC(V.sub(0), Constant(0.), both_ends)
    if bcs_case == "C-F":
        bc = [DirichletBC(V.sub(0), Constant(0.), left_end), DirichletBC(V.sub(1), Constant(0.), left_end)]
                     
    # shear locking problem
    dx_shear = dx(scheme="default",metadata={"quadrature_scheme":"default", "quadrature_degree": 1})
    
    f_st = f -dz_hh*ddf
    
    a = -(EI/Kf_hh)*(w.dx(0) - fi)*w_t.dx(0)*dx \
      + EI*fi.dx(0)*fi_t.dx(0)*dx  -  (EI/Kf_hh)*(w.dx(0) - fi)*fi_t*dx   

    L = -f*w_t*dx - dz_hh*df*w_t.dx(0)*dx \
     - dz_hh*df*fi_t*dx + Kf_hh*f_st*fi_t.dx(0)*dx - dz_hh*f*fi_t.dx(0)*dx
    
    A, b = assemble_system(a, L, bc)
    
    for i,d in enumerate(F_ext_cor):    
        ps = PointSource(V.sub(0), Point(F_ext_cor[i]), F_ext[i])
        ps.apply(b) #External forces
    
    u_h = Function(V)
    solve(A, u_h.vector(), b)
    
    Q_ = Function(W)
    M_ = Function(W)
    
    #  - dz_hh*df  - project(dz_hh*f ,W)   -project(u_h[1],W)
    M_ = project(-EI*(u_h.dx(0)[1]) - dz_hh*f + Kf_hh*f_st  ,W)
    Q_ = project( EI/(Kf_hh)*u_h.dx(0)[0] -  EI/(Kf_hh)*u_h[1] , W)
    
    return u_h, Q_, M_, mesh



def loads_(lds_case, q1):
    x_ = symbols('x[0]')
    
    if lds_case == 'sinusoidal': # constant value q1
        load_func = lambda x: q1*sin(math.pi*x/L_beam)
        load = q1*sp.sin(x_*math.pi/L_beam)
    if lds_case == 'const': # constant value q1
        load_func = lambda x: q1 + 0.0*x
        load = q1 + 0.0*x_
    if lds_case == 'linear': # from 0 to 11 linear
        load_func = lambda x: q1*1.0*x/L_beam
        load = q1*1.0*x_/L_beam
    if lds_case == 'quadratic':  # parabolic with max=q1 at  center
        load_func = lambda x: q1*(-4.0*x*x/(L_beam**2) + 4.0/L_beam*x)
        load = -4.0*x_*x_/(L_beam**2) + 4.0/L_beam*x_
    if lds_case == 'zero': # no loads
        load_func = lambda x: 0.0*x
        load = 0.0*x_
    # Derivative
    load_1 = load.diff(x_, 1)
    load_2 = load.diff(x_, 2)
    load_3 = load.diff(x_, 3)

    p_gen = load - dz_hh*load_2
    dp_gen = load_1 - dz_hh*load_3

    # Convert to Dolfin expression. 
    df = Expression(ccode(load_1), degree=2)
    ddf = Expression(ccode(load_2), degree=2)
    f = Expression(ccode(load), degree=2)
    f_star = Expression(ccode(p_gen), degree=2)
    df_star = Expression(ccode(dp_gen), degree=2)
    return [f,df,ddf, load_func, f_star,df_star]


#### Read BVPs solutions
mesh_CS = Mesh("temp_solutions/mesh.xml")
V = FunctionSpace(mesh_CS, 'DG', 1)
U_x_2, U_y_2, U_z_3, U_x_4, U_y_4, U_z_5 = [Function(V),Function(V),Function(V),Function(V),Function(V),Function(V)]
T_xx_2, T_xy_2, T_yy_2, T_xz_3, T_yz_3, T_zz_1 = [Function(V),Function(V),Function(V),Function(V),Function(V),Function(V)]
T_xx_4, T_xy_4, T_yy_4, T_xz_5, T_yz_5, T_zz_3 = [Function(V),Function(V),Function(V),Function(V),Function(V),Function(V)]

U_dict = [U_x_2, U_y_2, U_z_3, U_x_4, U_y_4, U_z_5]
T_dict = [T_xx_2, T_xy_2, T_yy_2, T_xz_3, T_yz_3, T_zz_1]
T_dict2 = [T_xx_4, T_xy_4, T_yy_4, T_xz_5, T_yz_5, T_zz_3]
for i,item in enumerate(U_dict + T_dict+T_dict2):
	var_name = ["U_x_2", "U_y_2", "U_z_3", "U_x_4", "U_y_4", "U_z_5"] +\
	["T_xx_2", "T_xy_2", "T_yy_2", "T_xz_3", "T_yz_3", "T_zz_1"] +\
	["T_xx_4", "T_xy_4", "T_yy_4", "T_xz_5", "T_yz_5", "T_zz_3"]
	sol_file = HDF5File(MPI.comm_world, "temp_solutions/"+ var_name[i]+ ".h5","r")
	sol_file.read(item,"/f")
	sol_file.close()

x = SpatialCoordinate(mesh_CS)
D1 = assemble(-x[0]*T_zz_1*dx(mesh_CS))
D2 = assemble(-x[0]*T_zz_3*dx(mesh_CS))
K_fi = assemble(-x[0]*U_z_3*dx(mesh_CS))/assemble(x[0]**2*dx(mesh_CS))

L_beam = 10.
h_beam = 1.

Kf_hh = K_fi * h_beam**2
dz_hh = D2/D1 * h_beam**2
EI = D1*h_beam**4
I = assemble(x[0]**2*dx(mesh_CS))*h_beam**4

D0 = FiniteElement("DG", interval, 0)
num_elems = 40
f,df,ddf,load_func, f_star, df_star = loads_('const', 1.0)

#F_cor = [L_beam/4,3*L_beam/4 ]
a_l = 76.2e-3
#F_v = 400.0
F_v = 0.0
F_cor = [a_l, L_beam - a_l ]
F_val = [-F_v, -F_v]
u_h,Q_,M_, mesh_1d = solver_GN(f,df,ddf, num_elems, 'C-C', F_cor , F_val )
w, fi_ = split(u_h)

num = 100
L_el = L_beam / num
plot_node = np.zeros(num+1)

plot_FENICS = np.zeros(num+1)
plot_FI = np.zeros(num+1)
plot_M = np.zeros(num+1)
plot_Q = np.zeros(num+1)
plot_load = np.zeros(num+1)
for i,d in enumerate(plot_node):
    plot_node[i] = i*L_el
    plot_FENICS[i] = float(u_h(i*L_el)[0])
    plot_FI[i] = float(u_h(i*L_el)[1])
    plot_M[i] = float(M_(i*L_el))
    plot_Q[i] = float(Q_(i*L_el))
    plot_load[i] = load_func(i * L_el)



#value = -5./384*L_beam**4/EI
value = -1./384*L_beam**4/EI
value2= L_beam**2/(8*I)
print("max_defl:", value, -plot_FENICS.max())
print("max_sigma_zz:", value2)

plt_res= False
if (plt_res == True):
	plt.subplot(4,1,1, ymargin = 0.5)
	plt.plot(plot_node, -plot_FENICS, label = 'deflection', color = 'black')
	plt.legend()
	plt.grid()
	
	plt.subplot(4,1,2)
	plt.plot(plot_node, -plot_FI, label = 'angle', color = 'black')
	plt.legend()
	plt.grid()
	
	plt.subplot(4,1,3)
	plt.plot(plot_node, -plot_M, label = 'Moment', color = 'black')
	plt.legend()
	plt.grid()
	
	plt.subplot(4,1,4)
	plt.plot(plot_node, -plot_Q, label = 'Force', color = 'black')
	plt.legend()
	plt.grid()
	
	plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9, hspace=0.5)
	
	plt.show()


z_val = L_beam/2
sigm_xz = -T_xz_3*h_beam**2*Q_(z_val)/EI + (T_xz_5*h_beam**4 - dz_hh*T_xz_3*h_beam**2)*(df(z_val))/EI
sigm_zz = -T_zz_1*h_beam*M_(z_val)/EI + (T_zz_3*h_beam**3 - dz_hh*T_zz_1*h_beam)*(f(z_val))/EI

u_x = u_h(z_val)[0] + U_x_2*(Kf_hh/EI*f_star(z_val) - u_h(z_val)[1]) + U_x_4*f_star(z_val)/EI
check = assemble(u_x*dx(mesh_CS))/assemble(1.*dx(mesh_CS))
print(check*assemble(1.*dx(mesh_CS)))
plot(mesh_CS)
c=plot(u_x, mesh = mesh_CS)
plt.colorbar(c)

plt.show()




