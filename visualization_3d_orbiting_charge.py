import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from scipy import optimize
import scipy.constants as const

eps = const.epsilon_0
pi = const.pi
e = const.e
c = const.c


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
    def retarded_time(self,tr,t,X,Y,Z):
        return ((X-self.xpos(tr))**2 + (Y-self.ypos(tr))**2 + (Z-self.zpos(tr))**2)**0.5 - c*(t-tr)


class OrbittingCharge(Charge):
    def __init__(self,pos_charge=True,phase=0,amplitude=2e-9,max_speed=0.9*c):
        super().__init__(pos_charge)
        self.amplitude=amplitude
        self.w=max_speed/amplitude
        self.phase=phase
    def xpos(self,t): return self.amplitude*np.cos(self.w*t+self.phase)
    def ypos(self,t): return self.amplitude*np.sin(self.w*t+self.phase)
    def zpos(self,t): return 0*t
    def xvel(self,t): return -self.amplitude*self.w*np.sin(self.w*t+self.phase)
    def yvel(self,t): return self.amplitude*self.w*np.cos(self.w*t+self.phase)
    def zvel(self,t): return 0*t
    def xacc(self,t): return -self.amplitude*self.w**2*np.cos(self.w*t+self.phase)
    def yacc(self,t): return -self.amplitude*self.w**2*np.sin(self.w*t+self.phase)
    def zacc(self,t): return 0*t


class MovingChargesField:
    def __init__(self,charges,h=1e-12):
        try: len(charges)
        except: charges=[charges]
        self.charges=charges
        self.h=h

    def _calculate_individual_E(self,charge,tr,X,Y,Z):
        rx = X - charge.xpos(tr)
        ry = Y - charge.ypos(tr)
        rz = Z - charge.zpos(tr)
        r = np.sqrt(rx**2+ry**2+rz**2)

        vx = charge.xvel(tr)
        vy = charge.yvel(tr)
        vz = charge.zvel(tr)

        ax = charge.xacc(tr)
        ay = charge.yacc(tr)
        az = charge.zacc(tr)

        ux = c*rx/r - vx
        uy = c*ry/r - vy
        uz = c*rz/r - vz

        r_dot_u = rx*ux + ry*uy + rz*uz
        r_dot_a = rx*ax + ry*ay + rz*az
        vmag2 = vx**2 + vy**2 + vz**2

        A = e/(4*pi*eps)*r/(r_dot_u**3)
        Ex_v = A*(c**2 - vmag2)*ux
        Ey_v = A*(c**2 - vmag2)*uy
        Ez_v = A*(c**2 - vmag2)*uz

        Ex_a = A*(r_dot_a*ux - r_dot_u*ax)
        Ey_a = A*(r_dot_a*uy - r_dot_u*ay)
        Ez_a = A*(r_dot_a*uz - r_dot_u*az)

        return Ex_v+Ex_a, Ey_v+Ey_a, Ez_v+Ez_a

    def calculate_E(self,t,X,Y,Z):
        tarr = np.ones_like(X)*t
        Ex = np.zeros_like(X)
        Ey = np.zeros_like(X)
        Ez = np.zeros_like(X)

        for charge in self.charges:
            tr = optimize.newton(charge.retarded_time, tarr, args=(tarr,X,Y,Z), tol=self.h)
            ex,ey,ez = self._calculate_individual_E(charge,tr,X,Y,Z)
            Ex += ex; Ey += ey; Ez += ez

        return Ex,Ey,Ez


R = 2e-9
beta = 0.99
charge = OrbittingCharge(True,0,R,beta*c)
field = MovingChargesField(charge)

GRID = 20
L = 6e-9

x = np.linspace(-L,L,GRID)
y = np.linspace(-L,L,GRID)
z = np.linspace(-L,L,GRID)
X3,Y3,Z3 = np.meshgrid(x,y,z,indexing="ij")

w0 = beta*c/R
T0 = 2*np.pi/w0
FRAMES = 80
times = np.linspace(0,2*T0,FRAMES)

fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(111,projection="3d")
ax.set_xlim(-L,L)
ax.set_ylim(-L,L)
ax.set_zlim(-L,L)
ax.set_box_aspect([1,1,1])

def update(i):
    ax.cla()

    ax.set_xlim(-L,L)
    ax.set_ylim(-L,L)
    ax.set_zlim(-L,L)
    ax.set_box_aspect([1,1,1])

    t = times[i]
    Ex,Ey,Ez = field.calculate_E(t,X3,Y3,Z3)

    # Clip extreme values
    m = np.percentile(np.sqrt(Ex**2 + Ey**2 + Ez**2), 95)
    Ex = np.clip(Ex, -m, m)
    Ey = np.clip(Ey, -m, m)
    Ez = np.clip(Ez, -m, m)

    SCALE = 1e-10
    Emag = np.sqrt(Ex**2 + Ey**2 + Ez**2)
    Emag_norm = (Emag - Emag.min()) / (Emag.max() - Emag.min() + 1e-30)
    cmap = plt.cm.inferno
    colors = cmap(Emag_norm)   

    Xf = X3.flatten()
    Yf = Y3.flatten()
    Zf = Z3.flatten()

    Exf = (Ex * SCALE).flatten()
    Eyf = (Ey * SCALE).flatten()
    Ezf = (Ez * SCALE).flatten()

    Cf = colors.reshape(-1,4)  
    ax.quiver(
        Xf, Yf, Zf,
        Exf, Eyf, Ezf,
        length=L/6,
        normalize=False,
        colors=Cf
    )
    cx = charge.xpos(t)
    cy = charge.ypos(t)
    ax.scatter([cx],[cy],[0], color="white", s=80, edgecolors="black")

    return []



ani = animation.FuncAnimation(fig, update, frames=FRAMES, interval=100)

ani.save("orbiting_charge_3D.gif", writer="pillow", fps=20)

plt.show()
