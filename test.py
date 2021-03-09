from dolfin import *
from matplotlib import  pyplot as plt
import CS_geometry

class BVPs:
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


	U_z_1 = Expression('-x[0]',degree=2)
	P2 = FiniteElement("Lagrange", triangle, 2)
	R = FiniteElement("R", triangle, 0)   

	
	def k_2(self):
		
		element = MixedElement([self.P2,self.P2,self.R,self.R, self.R, self.R])

		V_L = FunctionSpace(self.mesh, element)
		V_2 = TestFunction(V_L)
		V_x_2,V_y_2, test_lamb1, test_lamb2, test_const, test_lamb4 = split(V_2)
		
		U_2 = TrialFunction(V_L)
		U_x_2,U_y_2, lamb1, lamb2, const, lamb4 = split(U_2)

		## define linear froms
		## comutation will be after
		t_xx = self.Exxxx*U_x_2.dx(0) + self.Exxyy*U_y_2.dx(1) + self.Exxzz*(self.U_z_1 + const)
		t_yy = self.Exxyy*U_x_2.dx(0) + self.Eyyyy*U_y_2.dx(1) + self.Eyyzz*(self.U_z_1 + const)
		t_xy = 0.5*self.Gxy*(U_x_2.dx(1) + U_y_2.dx(0))
		t_zz = self.Exxzz*U_x_2.dx(0) + self.Eyyzz*U_y_2.dx(1) + self.Eyyzz*(self.U_z_1 + const)

		x_ = Expression('x[0]',degree=2)
		y_ = Expression('x[1]',degree=2)

		F = (t_xx*V_x_2.dx(0) + t_xy*V_x_2.dx(1) + t_xy*V_y_2.dx(0) + t_yy*V_y_2.dx(1))*dx
		F += (lamb1*V_x_2 + test_lamb1*U_x_2 + lamb2*V_y_2 + test_lamb2*U_y_2)*dx
		F += t_zz*test_const*dx  	
		F += (test_lamb4*(U_y_2*x_ - U_x_2*y_) + lamb4*(V_y_2*x_ - V_x_2*y_) )*dx  		
		## compute rhs and lhs
		a_ = lhs(F)
		L_ = rhs(F)

		U_2 = Function(V_L)
		solve(a_ == L_ , U_2, \
      		solver_parameters={"linear_solver": "lu", "preconditioner":"none"})
		(U_x_2, U_y_2, lamb1, lamb2, const, lamb4) = split(U_2)
		
		T_xx_2 = self.Exxxx*U_x_2.dx(0) + self.Exxyy*U_y_2.dx(1) + self.Exxzz*(self.U_z_1 + const)
		T_yy_2 = self.Exxyy*U_x_2.dx(0) + self.Eyyyy*U_y_2.dx(1) + self.Eyyzz*(self.U_z_1 + const)
		T_xy_2 = 0.5*self.Gxy*(U_x_2.dx(1) + U_y_2.dx(0))
		T_zz_1 = self.Exxzz*U_x_2.dx(0) + self.Eyyzz*U_y_2.dx(1) + self.Ezzzz*(self.U_z_1 + const)

		
		##### CHECKS, TESTS
		poisson = self.Exxyy/(2*(self.Exxyy+ 0.5*self.Gxy))
		U_x_ex = Expression('0.5*cof*(x[0]*x[0] - x[1]*x[1])',degree=2, cof=poisson)
		U_y_ex = Expression('cof*x[0]*x[1]',degree=2,cof=poisson)
		
		print("Int U_x_2 = 0: ", assemble(U_x_2*dx(self.mesh)))
		print("Int U_y_2 = 0: ", assemble(U_y_2*dx(self.mesh)))
		print("Int T_zz_1 = 0:", assemble(T_zz_1*dx(self.mesh)))
		
		print("U_x_2, L2 error: ", assemble((U_x_2 - U_x_ex)**2*dx(self.mesh))/assemble((U_x_ex)**2*dx(self.mesh)))
		print("U_y_2, L2 error: ", assemble((U_y_2 - U_y_ex)**2*dx(self.mesh))/assemble((U_y_ex)**2*dx(self.mesh)))

		return U_x_2, U_y_2, T_xx_2, T_yy_2, T_xy_2, T_zz_1, const

	def k_4(self, U_x_2_, U_y_2_, T_zz_1):
		T = FunctionSpace(self.mesh, self.P2)
		
		## solve second part for the first rpoblem, and first part for the second
		element = MixedElement([self.P2, self.P2, self.P2, self.R, self.R, self.R, self.R])

		V_L = FunctionSpace(self.mesh, element)
		V_4 = TestFunction(V_L)
		V_x_4,V_y_4, V_z_3, test_lamb1, test_lamb2, test_lamb3, test_lamb4 = split(V_4)
		
		U_4 = TrialFunction(V_L)
		U_x_4,U_y_4, U_z_3, lamb1, lamb2, lamb3, lamb4 = split(U_4)

		f_alpha = 1./assemble(1.*ds(1))
		print(f_alpha)

		x = SpatialCoordinate(self.mesh)
	

		poisson = self.Exxyy/(2*(self.Exxyy+ 0.5*self.Gxy))
		U_x_ex = Expression('0.5*cof*(x[0]*x[0] - x[1]*x[1])',degree=2, cof=poisson)
		U_y_ex = Expression('cof*x[0]*x[1]',degree=2,cof=poisson)


		## define linear froms
		## computation will be after
		t_xx = self.Exxxx*U_x_4.dx(0) + self.Exxyy*U_y_4.dx(1) + self.Exxzz*U_z_3
		t_yy = self.Exxyy*U_x_4.dx(0) + self.Eyyyy*U_y_4.dx(1) + self.Eyyzz*U_z_3
		t_zz = self.Exxzz*U_x_4.dx(0) + self.Eyyzz*U_y_4.dx(1) + self.Ezzzz*U_z_3
		t_xy = 0.5*self.Gxy*(U_x_4.dx(1) + U_y_4.dx(0))
		t_xz = 0.5*self.Gxz*(U_z_3.dx(0) +  U_x_2)
		t_yz = 0.5*self.Gyz*(U_z_3.dx(1) +  U_y_2)

		
		D1 = assemble(-x[0]*T_zz_1*dx(self.mesh))
		a_x = assemble(x[0]*dx(self.mesh))/assemble(1.0*dx(self.mesh))
		a_y = assemble(x[1]*dx(self.mesh))/assemble(1.0*dx(self.mesh))
		print("D1:", D1)

		x_ = Expression('x[0]',degree=2)
		y_ = Expression('x[1]',degree=2)

		F  = (t_xx*V_x_4.dx(0) + t_xy*V_x_4.dx(1) - t_xz*V_x_4)*dx - f_alpha*D1*V_x_4*ds(1)
		F += (t_xy*V_y_4.dx(0) + t_yy*V_y_4.dx(1) - t_yz*V_y_4)*dx
		F += (t_xz*V_z_3.dx(0) + t_yz*V_z_3.dx(1) - T_zz_1*V_z_3)*dx
		F += (lamb1*V_x_4 + test_lamb1*U_x_4 + lamb2*V_y_4 + test_lamb2*U_y_4)*dx
		F += (lamb3*(self.Exxzz*V_x_4.dx(0) + self.Eyyzz*V_y_4.dx(1) + self.Ezzzz*V_z_3) + test_lamb3*t_zz)*dx
		F += (test_lamb4*(U_y_4*x_ - U_x_4*y_) + lamb4*(V_y_4*x_ - V_x_4*y_) )*dx  	
		## compute rhs and lhs
		a_ = lhs(F)
		L_ = rhs(F)

		U_4 = Function(V_L)

		#solver = KrylovSolver("gmres", "icc")
		#solver.parameters["relative_tolerance"] = 5e-6
		#solver.parameters["maximum_iterations"] = 1000
		#solver.parameters["monitor_convergence"] = True
		
		#problem = LinearVariationalProblem(a_ , L_ , U_4)
		#solver = LinearVariationalSolver(problem)
		#solver.solve()
		solve(a_ == L_ , U_4, \
      		solver_parameters={"linear_solver": "lu", "preconditioner":"none"})
		(U_x_4, U_y_4, U_z_3, lamb1, lamb2, lamb3, lamb4) = split(U_4)

		
		'''
		amg              |  Algebraic multigrid                       
		default          |  default preconditioner                    
		hypre_amg        |  Hypre algebraic multigrid (BoomerAMG)     
		hypre_euclid     |  Hypre parallel incomplete LU factorization
		hypre_parasails  |  Hypre parallel sparse approximate inverse 
		icc              |  Incomplete Cholesky factorization         
		ilu              |  Incomplete LU factorization               
		jacobi           |  Jacobi iteration                          
		none             |  No preconditioner                         
		petsc_amg        |  PETSc algebraic multigrid                 
		sor              |  Successive over-relaxation
		'''
		
		T_xx_4 = self.Exxxx*U_x_4.dx(0) + self.Exxyy*U_y_4.dx(1) + self.Exxzz*(U_z_3)
		T_yy_4 = self.Exxyy*U_x_4.dx(0) + self.Eyyyy*U_y_4.dx(1) + self.Eyyzz*(U_z_3)
		T_xy_4 = 0.5*self.Gxy*(U_x_4.dx(1) + U_y_4.dx(0))
		T_zz_3 = self.Exxzz*U_x_4.dx(0) + self.Eyyzz*U_y_4.dx(1) + self.Ezzzz*(U_z_3)
		T_xz_3 = 0.5*self.Gxz*(U_z_3.dx(0) + U_x_2_)
		T_yz_3 = 0.5*self.Gyz*(U_z_3.dx(1) + U_y_2_)
		

		##### CHECKS, TESTS
		print("Int U_x_4 = 0: ", assemble(U_x_4*dx(self.mesh)))
		print("Int U_y_4 = 0: ", assemble(U_y_4*dx(self.mesh)))
		print("Int T_zz_3 = 0:", assemble(T_zz_3*dx(self.mesh)))
		
		return U_x_4, U_y_4, U_z_3, T_xx_4, T_yy_4, T_xy_4, T_zz_3, T_xz_3, T_yz_3 

	def k_6(self, U_x_4_, U_y_4_, T_zz_3):
		T = FunctionSpace(self.mesh, self.P2)
		
		
		## solve second part for the first rpoblem, and first part for the second
		element = MixedElement([self.P2, self.R])

		V_L = FunctionSpace(self.mesh, element)
		V_4 = TestFunction(V_L)
		V_z_5, test_lamb3 = split(V_4)
		
		U_4 = TrialFunction(V_L)
		U_z_5, lamb3 = split(U_4)


		## define linear froms
		## computation will be after
		t_xz = 0.5*self.Gxz*(U_z_5.dx(0) +  U_x_4_)
		t_yz = 0.5*self.Gyz*(U_z_5.dx(1) +  U_y_4_)

		
		F = (t_xz*V_z_5.dx(0) + t_yz*V_z_5.dx(1) - T_zz_3*V_z_5)*dx
		F += (lamb3*V_z_5 + test_lamb3*U_z_5)*dx
		## compute rhs and lhs
		a_ = lhs(F)
		L_ = rhs(F)

		U_4 = Function(V_L)

		#solver = KrylovSolver("gmres", "icc")
		#solver.parameters["relative_tolerance"] = 5e-6
		#solver.parameters["maximum_iterations"] = 1000
		#solver.parameters["monitor_convergence"] = True
		
		#problem = LinearVariationalProblem(a_ , L_ , U_4)
		#solver = LinearVariationalSolver(problem)
		#solver.solve()
		solve(a_ == L_ , U_4, \
      		solver_parameters={"linear_solver": "lu", "preconditioner":"none"})
		(U_z_5, lamb3) = split(U_4)

		
		'''
		amg              |  Algebraic multigrid                       
		default          |  default preconditioner                    
		hypre_amg        |  Hypre algebraic multigrid (BoomerAMG)     
		hypre_euclid     |  Hypre parallel incomplete LU factorization
		hypre_parasails  |  Hypre parallel sparse approximate inverse 
		icc              |  Incomplete Cholesky factorization         
		ilu              |  Incomplete LU factorization               
		jacobi           |  Jacobi iteration                          
		none             |  No preconditioner                         
		petsc_amg        |  PETSc algebraic multigrid                 
		sor              |  Successive over-relaxation
		'''
		
		T_xz_5 = 0.5*self.Gxz*(U_z_5.dx(0) + U_x_4_)
		T_yz_5 = 0.5*self.Gyz*(U_z_5.dx(1) + U_y_4_)
		

		#c = plot(T_xz_3, mesh=self.mesh)
		#plot(self.mesh)
		#plt.colorbar(c)
		#plt.show() 


		##### CHECKS, TESTS
		print("Int U_z_5 = 0: ", assemble(U_z_5*dx(self.mesh)))
		
		return  U_z_5, T_xz_5, T_yz_5

	def k_alt(self, U_x_2_, U_y_2_, T_zz_1):
		T = FunctionSpace(self.mesh, self.P2)
		
		
		## solve second part for the first rpoblem, and first part for the second
		element = MixedElement([self.P2, self.R])

		V_L = FunctionSpace(self.mesh, element)
		V_4 = TestFunction(V_L)
		V_z_3, test_lamb3 = split(V_4)
		
		U_4 = TrialFunction(V_L)
		U_z_3, lamb3 = split(U_4)


		## define linear froms
		## computation will be after
		t_xz = 0.5*self.Gxz*(U_z_3.dx(0) +  U_x_2)
		t_yz = 0.5*self.Gyz*(U_z_3.dx(1) +  U_y_2)

		
		F = (t_xz*V_z_3.dx(0) + t_yz*V_z_3.dx(1) - T_zz_1*V_z_3)*dx
		F += (lamb3*V_z_3 + test_lamb3*U_z_3)*dx
		## compute rhs and lhs
		a_ = lhs(F)
		L_ = rhs(F)

		U_4 = Function(V_L)

		#solver = KrylovSolver("gmres", "icc")
		#solver.parameters["relative_tolerance"] = 5e-6
		#solver.parameters["maximum_iterations"] = 1000
		#solver.parameters["monitor_convergence"] = True
		
		#problem = LinearVariationalProblem(a_ , L_ , U_4)
		#solver = LinearVariationalSolver(problem)
		#solver.solve()
		solve(a_ == L_ , U_4, \
      		solver_parameters={"linear_solver": "lu", "preconditioner":"none"})
		(U_z_3, lamb3) = split(U_4)

		
		'''
		amg              |  Algebraic multigrid                       
		default          |  default preconditioner                    
		hypre_amg        |  Hypre algebraic multigrid (BoomerAMG)     
		hypre_euclid     |  Hypre parallel incomplete LU factorization
		hypre_parasails  |  Hypre parallel sparse approximate inverse 
		icc              |  Incomplete Cholesky factorization         
		ilu              |  Incomplete LU factorization               
		jacobi           |  Jacobi iteration                          
		none             |  No preconditioner                         
		petsc_amg        |  PETSc algebraic multigrid                 
		sor              |  Successive over-relaxation
		'''
		
		T_xz_3 = 0.5*self.Gxz*(U_z_3.dx(0) + U_x_2_)
		T_yz_3 = 0.5*self.Gyz*(U_z_3.dx(1) + U_y_2_)
		

		#c = plot(T_xz_3, mesh=self.mesh)
		#plot(self.mesh)
		#plt.colorbar(c)
		#plt.show() 


		##### CHECKS, TESTS
		print("Int U_z_3 = 0: ", assemble(U_z_3*dx(self.mesh)))
		
		return  U_z_3, T_xz_3, T_yz_3

	def k_alt2(self, U_z_3_, T_xz_3_, T_yz_3_, T_zz_1_):
		
		T = FunctionSpace(self.mesh, self.P2)
		
		class Top(SubDomain):
		    def inside(self, x, on_boundary):
        		return near(x[0], -0.5) and on_boundary

		## solve second part for the first rpoblem, and first part for the second
		element = MixedElement([self.P2, self.P2, self.R, self.R, self.R])

		V_L = FunctionSpace(self.mesh, element)
		V_4 = TestFunction(V_L)
		V_x_4,V_y_4, test_lamb1, test_lamb2, test_const_U_z_3 = split(V_4)
		
		U_4 = TrialFunction(V_L)
		U_x_4,U_y_4, lamb1, lamb2, const_Uz_3 = split(U_4)

		top = Top()
		boundaries = MeshFunction('size_t', self.mesh, dim = 1)
		boundaries.set_all(0)
		top.mark(boundaries, 1)
		ds = Measure('ds', domain=self.mesh, subdomain_data=boundaries)
		x = SpatialCoordinate(self.mesh)
		f_alpha = 1./1.

		## define linear froms
		## computation will be after
		t_xx = self.Exxxx*U_x_4.dx(0) + self.Exxyy*U_y_4.dx(1) + self.Exxzz*(U_z_3_ + const_Uz_3)
		t_yy = self.Exxyy*U_x_4.dx(0) + self.Eyyyy*U_y_4.dx(1) + self.Eyyzz*(U_z_3_ + const_Uz_3)
		t_zz = self.Exxzz*U_x_4.dx(0) + self.Eyyzz*U_y_4.dx(1) + self.Ezzzz*(U_z_3_ + const_Uz_3)
		t_xy = 0.5*self.Gxy*(U_x_4.dx(1) + U_y_4.dx(0))

		
		D1 = assemble(-x[0]*T_zz_1_*dx(self.mesh))
		print("D1:", D1)

		F  = (t_xx*V_x_4.dx(0) + t_xy*V_x_4.dx(1) - T_xz_3_*V_x_4)*dx #- f_alpha*D1*V_x_4*ds(1)
		F += (t_xy*V_y_4.dx(0) + t_yy*V_y_4.dx(1) - T_yz_3_*V_y_4)*dx
		F += (lamb1*V_x_4 + test_lamb1*U_x_4 + lamb2*V_y_4 + test_lamb2*U_y_4)*dx
		F += (t_zz*test_const_U_z_3)*dx  	
		## compute rhs and lhs
		a_ = lhs(F)
		L_ = rhs(F)

		U_4 = Function(V_L)

		#solver = KrylovSolver("gmres", "icc")
		#solver.parameters["relative_tolerance"] = 5e-6
		#solver.parameters["maximum_iterations"] = 1000
		#solver.parameters["monitor_convergence"] = True
		
		#problem = LinearVariationalProblem(a_ , L_ , U_4)
		#solver = LinearVariationalSolver(problem)
		#solver.solve()
		solve(a_ == L_ , U_4, \
      		solver_parameters={"linear_solver": "lu", "preconditioner":"icc"})
		(U_x_4, U_y_4, lamb1, lamb2, const_Uz_3) = split(U_4)

		
		'''
		amg              |  Algebraic multigrid                       
		default          |  default preconditioner                    
		hypre_amg        |  Hypre algebraic multigrid (BoomerAMG)     
		hypre_euclid     |  Hypre parallel incomplete LU factorization
		hypre_parasails  |  Hypre parallel sparse approximate inverse 
		icc              |  Incomplete Cholesky factorization         
		ilu              |  Incomplete LU factorization               
		jacobi           |  Jacobi iteration                          
		none             |  No preconditioner                         
		petsc_amg        |  PETSc algebraic multigrid                 
		sor              |  Successive over-relaxation
		'''
		
		T_xx_4 = self.Exxxx*U_x_4.dx(0) + self.Exxyy*U_y_4.dx(1) + self.Exxzz*(U_z_3 + const_Uz_3)
		T_yy_4 = self.Exxyy*U_x_4.dx(0) + self.Eyyyy*U_y_4.dx(1) + self.Eyyzz*(U_z_3 + const_Uz_3)
		T_xy_4 = 0.5*self.Gxy*(U_x_4.dx(1) + U_y_4.dx(0))
		T_zz_3 = self.Exxzz*U_x_4.dx(0) + self.Eyyzz*U_y_4.dx(1) + self.Ezzzz*(U_z_3 + const_Uz_3)

		c = plot(T_xz_3, mesh=mesh)
		plt.colorbar(c)
		plot(self.mesh)
		plt.show() 


		##### CHECKS, TESTS
		print("Int U_x_4 = 0: ", assemble(U_x_4*dx(self.mesh)))
		print("Int U_y_4 = 0: ", assemble(U_y_4*dx(self.mesh)))
		print("Int T_zz_3 = 0:", assemble(T_zz_3*dx(self.mesh)))
		
		return U_x_4,  U_y_4, T_xx_4, T_yy_4, T_xy_4, T_zz_3, const_Uz_3 
 


class Top(SubDomain):
	#def __init__(self, f_x):
	#	self.f_x = f_x
	def inside(self, x, on_boundary):
		return near(x[0], -0.5) and (abs(x[1])<0.1) and on_boundary


if __name__ == '__main__':
	L = 10.
	rad = 1.
	thick = 0.1
	f_x = 0.1
	#mesh = RectangleMesh(Point(-0.5,-0.5), Point(0.5,0.5) , 8, 8, diagonal='crossed')
	#mesh = CS_geometry.mesh_T_profile(0.5, thick*0.5/rad, 32)
	#mesh = CS_geometry.mesh_hollow_cylinder(0.5, thick*0.5/rad, 16)
	mesh = CS_geometry.mesh_hollow_rectangle(0.5, thick*0.5/rad, 32)

	top = Top()
	boundaries = MeshFunction('size_t', mesh, dim = 1)
	boundaries.set_all(0)
	top.mark(boundaries, 1)
	ds = Measure('ds', domain=mesh, subdomain_data=boundaries)
		
	E_ = [1.0,1.0,1.0,0.7,0.7,0.7]
	G_ = [0.3,0.3,0.3]
	
	Problem = BVPs(E_,G_, mesh)
	
	T = FunctionSpace(mesh, "DG", 1)
	[U_x_2, U_y_2, T_xx_2, T_yy_2, T_xy_2, T_zz_1, const] = Problem.k_2()  
	[U_x_4, U_y_4, U_z_3, T_xx_4, T_yy_4, T_xy_4, T_zz_3, T_xz_3, T_yz_3] = Problem.k_4(U_x_2, U_y_2, T_zz_1) 
	[U_z_5, T_xz_5, T_yz_5] = Problem.k_6(U_x_4, U_y_4, T_zz_3)
	#[U_z_3, T_xz_3, T_yz_3] = Problem.k_alt(cur_1, cur_2, T_zz_1)
	#[U_x_4,  U_y_4, T_xx_4, T_yy_4, T_xy_4, T_zz_3, const_Uz_3] = Problem.k_alt2(U_z_3, T_xz_3, T_yz_3, T_zz_1)


	plot(mesh)
	c = plot(T_xz_5, mesh=mesh)
	#print(project(U_x_2,T).vector().min())
	plt.colorbar(c)
	plt.show()


		#sol_file1 = File("temp_solutions/u1x_saved.pvd")
	#sol_file1 << project(U_x_2, T)
	U_dict = [U_x_2, U_y_2, U_z_3, U_x_4, U_y_4, U_z_5]
	T_dict = [T_xx_2, T_xy_2, T_yy_2, T_xz_3, T_yz_3, T_zz_1]
	T_dict2 = [T_xx_4, T_xy_4, T_yy_4, T_xz_5, T_yz_5, T_zz_3]
	for i,item in enumerate(U_dict + T_dict + T_dict2):
		var_name = ["U_x_2", "U_y_2", "U_z_3", "U_x_4", "U_y_4", "U_z_5"] +\
		["T_xx_2", "T_xy_2", "T_yy_2", "T_xz_3", "T_yz_3", "T_zz_1"] +\
		["T_xx_4", "T_xy_4", "T_yy_4", "T_xz_5", "T_yz_5", "T_zz_3"]
		sol_file = HDF5File(MPI.comm_world, "temp_solutions/"+ var_name[i]+ ".h5","w")
		sol_file.write(project(item,T),"/f")
		sol_file.close()

	mesh_file = File("temp_solutions/mesh.xml")
	mesh_file << mesh
 



'''
bicgstab       |  Biconjugate gradient stabilized method                      
cg             |  Conjugate gradient method                                   
default        |  default linear solver                                       
gmres          |  Generalized minimal residual method                         
minres         |  Minimal residual method                                     
mumps          |  MUMPS (MUltifrontal Massively Parallel Sparse direct Solver)
petsc          |  PETSc built in LU solver                                    
richardson     |  Richardson method                                           
superlu        |  SuperLU                                                     
superlu_dist   |  Parallel SuperLU                                            
tfqmr          |  Transpose-free quasi-minimal residual method                
umfpack        |  UMFPACK (Unsymmetric MultiFrontal sparse LU factorization) 
'''