from __future__ import print_function
from dolfin import *
import ufl
import numpy as np

parameters["form_compiler"]["representation"] = "uflacs"
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["quadrature_degree"] = 2

# ********** Time constants ********* #

t = 0.0; dt = 100.0; Tfinal=10000; freqSave = 10; freqMech = 5; inc = 0

# ******* Create mesh and define function spaces **********

# Mesh original : 64, 64, 3 
mesh = BoxMesh(Point(0,0,0),Point(50.0,50.0,2.5),95,95,3) #3.0 4
nn = FacetNormal(mesh)
bdry = MeshFunction("size_t", mesh, 2)
bdry.set_all(0)

# ******* Output file 
fileO = XDMFFile(mesh.mpi_comm(), "outputs/NH-RD-growth-Fig11-MSet2.xdmf")
fileO.parameters['rewrite_function_mesh'] = False
fileO.parameters["functions_share_mesh"] = True
fileO.parameters["flush_output"] = True

# ********* FE spaces ********* #
P1   = FiniteElement("CG", mesh.ufl_cell(), 1)
P1dc = FiniteElement("CG", mesh.ufl_cell(), 1) #CG or DG!!
P2v  = VectorElement("CG", mesh.ufl_cell(), 2)

Mh = FunctionSpace(mesh, P1)
Nh = FunctionSpace(mesh, MixedElement([P1,P1]))
Hh = FunctionSpace(mesh,P2v*P1dc)
Vh = FunctionSpace(mesh, P2v)

print (" **************** Mech Dof = ", Hh.dim())
print (" **************** Electr Dof = ", Nh.dim())

Sol    = Function(Nh)
(w1,w2)   = TrialFunctions(Nh)
(s1,s2)  = TestFunctions(Nh)

up      = Function(Hh)
dup     = TrialFunction(Hh)
(u,p)   = split(up)
(uT,pT) = TestFunctions(Hh)

uold = Function(Vh)
uuold = Function(Vh)

# ********* Mechanical parameters ********* #

mu    = Constant(4.0)
rho   = Constant(1.0)
robin = Constant(0.001)
stabP = Constant(0.1)

f0 = Constant((1,0,0))
s0 = Constant((0,1,0))
n0 = Constant((0,0,1))

# ********* Chemical parameters ********* #

D1 = Constant(0.005)
D2 = Constant(0.2)
mu1  = Constant(0.005)
mu2  = Constant(0.015)
rho1 = Constant(0.006)#0.002,0.006,0.01,0.02
rho2 = Constant(0.05)
Kh   = Constant(0.01)
kappa= Constant(0.1)

# ********* Initial data **** #

w1old = Function(Mh)
pert  = np.random.uniform(-1,1,w1old.vector().get_local().size)

eta   = Constant(mu1 * rho2 / ( mu2 * rho1 ))
w20 = Constant(0.406486)#0.0237564,0.406486,1.13891,4.1783
w10 = Constant(w20/(eta*(Kh+w20)))

w1old.vector()[:] = pert*0.1*w10 + w10
w2old = interpolate(w20,Mh)

# ********* Boundaries and boundary conditions ******* #

Bottom = CompiledSubDomain("near(x[2],0) && on_boundary")
Remain = CompiledSubDomain("(x[2] > 0) && on_boundary")
Bottom.mark(bdry,91); Remain.mark(bdry,92)
ds = Measure("ds", subdomain_data=bdry)

bcU = DirichletBC(Hh.sub(0).sub(2), Constant(0.0), bdry, 91)

# ********* Coupling constants ********* #

creact = Constant(0.0001)

# ********* Growth functions ******* #
gr0 = Constant(0.15)#0.1
gr1 = Constant(1.5)#0.5
gfs = Expression("gr0*t/Tfinal", t=t, Tfinal=Tfinal, gr0=gr0, degree=2)
gn  = Expression("gr1*w2*t/Tfinal*x[2]", w2 = w2old, t=t, Tfinal=Tfinal, gr1=gr1, degree=2)

# ******** Mechanical entities ************* #

ndim = u.geometric_dimension()
I = Identity(ndim); F = I + grad(u); F = variable(F)
C = F.T*F; B = F*F.T; J = det(F); invF = inv(F);
I1 = tr(C); stretch = pow(dot(F*f0,F*f0),0.5)

F_a = I+ gfs*outer(f0,f0) + gfs*outer(s0,s0) + gn * outer(n0,n0)
invFa = inv(F_a)
F_e = variable(F * invFa); B_e = F_e*F_e.T
J_e = det(F_e)

Piola = J * (mu*B_e - p*I) * invF.T 

FF = rho / pow(dt,2.0) * dot(u-2.0*uold+uuold,uT) * dx \
     + inner(Piola, grad(uT)) * dx + pT*(J_e-1.0)*dx \
     + stabP * dot(grad(p),grad(pT)) * dx \
     + dot(J*robin*invF.T*u,uT) * ds(92)

Tang = derivative(FF, up, dup)

# ***** Diffusion ***** #

KL = w1/dt * s1 * dx + inner(D1*J*inv(C)*grad(w1),grad(s1))*dx \
     + w2/dt * s2 * dx + inner(D2*J*inv(C)*grad(w2),grad(s2))*dx

KR = w1old/dt * s1 * dx \
     + (rho1*pow(w1old,2.0)/((1+kappa*pow(w1old,2.0))*(Kh+w2old))-mu1*w1old)*s1*dx \
     + w2old/dt*s2*dx + (rho2*pow(w1old,2.0)/(1+kappa*pow(w1old,2.0))-mu2*w2old)*s2*dx \
     - creact * I1 * s2 * dx 

# ************* Time loop ********
while (t <= Tfinal):
    
    print("t=%.2f" % t)
    
    RHS = assemble(KR); LHS = assemble(KL)
    solve(LHS,Sol.vector(),RHS)
    w1,w2 = Sol.split()
    assign(w1old,w1); assign(w2old,w2)
    
    if (inc % freqMech == 0):

        gfs.t = t; gn.t=t; gn.w2=w2old
        
        solve(FF == 0, up, J=Tang, bcs = bcU, 
              solver_parameters={'newton_solver':{'linear_solver':'mumps',\
                                                  'absolute_tolerance':5.0e-4,\
                                                  'relative_tolerance':5.0e-4,\
                                                  'maximum_iterations':30}})

        (u,p)=up.split()
        assign(uold,u); assign(uuold,uold)
    
        if (inc % freqSave == 0):
            u.rename("u","u"); fileO.write(u,t)
            p.rename("p","p"); fileO.write(p,t)
            w1.rename("w1","w1"); fileO.write(w1,t)
            w2.rename("w2","w2"); fileO.write(w2,t)
            
    t += dt; inc += 1

# ************* End **************** #
