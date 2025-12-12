import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from scipy import optimize
import scipy.constants as const

eps = const.epsilon_0
pi  = const.pi
e   = const.e
c   = const.c

class Charge:
    def __init__(self, pos_charge=True):
        self.pos_charge = pos_charge

    def xpos(self,t): pass
    def ypos(self,t): pass
    def zpos(self,t): pass
    def xvel(self,t): pass
    def yvel(self,t): pass
    def zvel(self,t): pass
    def xacc(self,t): pass
    def yacc(self,t): pass
    def zacc(self,t): pass

    def retarded_time(self, tr, t, X, Y, Z):
        return np.sqrt((X-self.xpos(tr))**2 +
                       (Y-self.ypos(tr))**2 +
                       (Z-self.zpos(tr))**2) - c*(t - tr)

class OscillatingCharge(Charge):

    def __init__(self, amplitude=2e-9, max_speed=0.9*c, pos_charge=True):
        super().__init__(pos_charge)
        self.A = amplitude
        self.w = max_speed / amplitude   
    def xpos(self,t): return self.A * np.cos(self.w*t)
    def ypos(self,t): return 0*t
    def zpos(self,t): return 0*t

    def xvel(self,t): return -self.A*self.w*np.sin(self.w*t)
    def yvel(self,t): return 0*t
    def zvel(self,t): return 0*t

    def xacc(self,t): return -self.A*self.w**2*np.cos(self.w*t)
    def yacc(self,t): return 0*t
    def zacc(self,t): return 0*t

class MovingChargesField:

    def __init__(self, charges, h=1e-12):
        try: len(charges)
        except: charges=[charges]
        self.charges = charges
        self.h = h

    def _calculate_individual_E(self, charge, tr, X, Y, Z):

        rx = X - charge.xpos(tr)
        ry = Y - charge.ypos(tr)
        rz = Z - charge.zpos(tr)
        r  = np.sqrt(rx**2 + ry**2 + rz**2)

        vx,vy,vz = charge.xvel(tr), charge.yvel(tr), charge.zvel(tr)
        ax,ay,az = charge.xacc(tr), charge.yacc(tr), charge.zacc(tr)

        ux = c*rx/r - vx
        uy = c*ry/r - vy
        uz = c*rz/r - vz

        r_dot_u = rx*ux + ry*uy + rz*uz
        r_dot_a = rx*ax + ry*ay + rz*az
        v2 = vx**2 + vy**2 + vz**2

        A = e/(4*pi*eps) * r/(r_dot_u**3)

        Ex_v = A*(c**2 - v2)*ux
        Ey_v = A*(c**2 - v2)*uy
        Ez_v = A*(c**2 - v2)*uz

        Ex_a = A*(r_dot_a*ux - r_dot_u*ax)
        Ey_a = A*(r_dot_a*uy - r_dot_u*ay)
        Ez_a = A*(r_dot_a*uz - r_dot_u*az)

        return Ex_v+Ex_a, Ey_v+Ey_a, Ez_v+Ez_a

    def calculate_E(self, t, X, Y, Z):
        tarr = np.ones_like(X)*t
        Ex = np.zeros_like(X)
        Ey = np.zeros_like(X)
        Ez = np.zeros_like(X)

        for charge in self.charges:
            tr = optimize.newton(charge.retarded_time,
                                 tarr, args=(tarr,X,Y,Z), tol=self.h)
            ex,ey,ez = self._calculate_individual_E(charge,tr,X,Y,Z)
            Ex+=ex; Ey+=ey; Ez+=ez

        return Ex,Ey,Ez

charge = OscillatingCharge()
field  = MovingChargesField(charge)

GRID = 20
L = 6e-9

x = np.linspace(-L,L,GRID)
y = np.linspace(-L,L,GRID)
z = np.linspace(-L,L,GRID)

X3,Y3,Z3 = np.meshgrid(x,y,z,indexing="ij")

FRAMES = 100
Tmax = 4*np.pi/charge.w    
times = np.linspace(0, Tmax, FRAMES)

fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(111,projection="3d")
ax.set_xlim(-L,L)
ax.set_ylim(-L,L)
ax.set_zlim(-L,L)
ax.set_box_aspect([1,1,1])

quiver_obj = None
charge_marker = ax.scatter([],[],[], color="white", s=80, edgecolors="black")

def update(i):
    global quiver_obj

    if quiver_obj is not None:
        quiver_obj.remove()

    t = times[i]
    Ex,Ey,Ez = field.calculate_E(t,X3,Y3,Z3)

    m = np.percentile(np.sqrt(Ex**2+Ey**2+Ez**2),95)
    Ex = np.clip(Ex,-m,m)
    Ey = np.clip(Ey,-m,m)
    Ez = np.clip(Ez,-m,m)

    SCALE = 1e-9

    Emag = np.sqrt(Ex**2 + Ey**2 + Ez**2)
    Emag_norm = (Emag - Emag.min())/(Emag.max() - Emag.min() + 1e-30)
    colors = plt.cm.inferno(Emag_norm).reshape(-1,4)

    Xf,Yf,Zf = X3.flatten(),Y3.flatten(),Z3.flatten()
    Exf,Eyf,Ezf = (Ex*SCALE).flatten(), (Ey*SCALE).flatten(), (Ez*SCALE).flatten()

    quiver_obj = ax.quiver(
        Xf,Yf,Zf,
        Exf,Eyf,Ezf,
        length=L/6,
        normalize=False,
        colors=colors
    )

    cx = charge.xpos(t)
    cy = 0
    cz = 0
    charge_marker._offsets3d = ([cx],[cy],[cz])

    return []

ani = animation.FuncAnimation(fig, update, frames=FRAMES, interval=80)
ani.save("Efield_OscillatingCharge.gif", writer="pillow", fps=20)

plt.show()
