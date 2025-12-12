import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy import optimize
import scipy.constants as const

eps = const.epsilon_0
mu = const.mu_0
pi = const.pi
e = const.e
c = const.c


class Charge:
    def __init__(self, pos_charge=True):
        self.pos_charge = pos_charge
    def xpos(self, t): pass
    def ypos(self, t): pass
    def zpos(self, t): pass
    def xvel(self, t): pass
    def yvel(self, t): pass
    def zvel(self, t): pass
    def xacc(self, t): pass
    def yacc(self, t): pass
    def zacc(self, t): pass
    def retarded_time(self, tr, t, X, Y, Z):
        return ((X - self.xpos(tr))**2 + (Y - self.ypos(tr))**2 + (Z - self.zpos(tr))**2)**0.5 - c*(t - tr)


class OrbittingCharge(Charge):
    def __init__(self, pos_charge=True, phase=0, amplitude=2e-9, max_speed=0.9*c):
        super().__init__(pos_charge)
        self.amplitude = amplitude
        self.w = max_speed / amplitude
        self.phase = phase
    def xpos(self, t): return self.amplitude*np.cos(self.w*t + self.phase)
    def ypos(self, t): return self.amplitude*np.sin(self.w*t + self.phase)
    def zpos(self, t): return 0*t
    def xvel(self, t): return -self.amplitude*self.w*np.sin(self.w*t + self.phase)
    def yvel(self, t): return self.amplitude*self.w*np.cos(self.w*t + self.phase)
    def zvel(self, t): return 0*t
    def xacc(self, t): return -self.amplitude*self.w**2*np.cos(self.w*t + self.phase)
    def yacc(self, t): return -self.amplitude*self.w**2*np.sin(self.w*t + self.phase)
    def zacc(self, t): return 0*t


class MovingChargesField:
    def __init__(self, charges, h=1e-12):
        try: len(charges)
        except: charges=[charges]
        self.charges = charges
        self.h = h

    def _calculate_individual_E(self, charge, tr, X, Y, Z, mode):
        rx = X - charge.xpos(tr)
        ry = Y - charge.ypos(tr)
        rz = Z - charge.zpos(tr)
        r = (rx**2 + ry**2 + rz**2)**0.5
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
        const0 = e/(4*pi*eps) * r/(r_dot_u**3)
        if not charge.pos_charge: const0 = -const0
        Ex_v = const0*(c**2 - vmag2)*ux
        Ey_v = const0*(c**2 - vmag2)*uy
        Ez_v = const0*(c**2 - vmag2)*uz
        Ex_a = const0*(r_dot_a*ux - r_dot_u*ax)
        Ey_a = const0*(r_dot_a*uy - r_dot_u*ay)
        Ez_a = const0*(r_dot_a*uz - r_dot_u*az)
        if mode=='Velocity': return Ex_v,Ey_v,Ez_v
        if mode=='Acceleration': return Ex_a,Ey_a,Ez_a
        return Ex_v+Ex_a, Ey_v+Ey_a, Ez_v+Ez_a

    def calculate_E(self, t, X, Y, Z, mode='Total', plane=False):
        tarr = np.ones_like(X)*t
        Ex = np.zeros_like(X); Ey = np.zeros_like(X); Ez = np.zeros_like(X)
        for charge in self.charges:
            tr = optimize.newton(charge.retarded_time, tarr, args=(tarr,X,Y,Z), tol=self.h)
            ex,ey,ez = self._calculate_individual_E(charge,tr,X,Y,Z,mode)
            Ex+=ex; Ey+=ey; Ez+=ez
        if plane: return Ex[:,:,0],Ey[:,:,0],Ez[:,:,0]
        return Ex,Ey,Ez


R = 2e-9
beta=0.99
charge = OrbittingCharge(True,0,R,beta*c)
field_engine = MovingChargesField(charge)

GRID_N = 40
L = 6e-9
x = np.linspace(-L, L, GRID_N)
y = np.linspace(-L, L, GRID_N)
X2,Y2 = np.meshgrid(x,y,indexing='xy')
Z2 = np.zeros_like(X2)
X = X2[:,:,None]
Y = Y2[:,:,None]
Z = Z2[:,:,None]

w0 = beta*c / R
T0 = 2*np.pi / w0
FRAMES = 200
times = np.linspace(0,2*T0,FRAMES)

fig,ax = plt.subplots(figsize=(7,7))
ax.set_xlim(-L,L)
ax.set_ylim(-L,L)
ax.set_aspect('equal')

im = ax.imshow(np.zeros_like(X2),origin='lower',extent=[-L,L,-L,L],cmap='inferno')

ds = 2
Qx = X2[::ds,::ds]
Qy = Y2[::ds,::ds]

Q = ax.quiver(
    Qx, Qy,
    np.zeros_like(Qx), np.zeros_like(Qy),
    color='white',
    scale_units='xy',
    scale=5e19,
    width=0.001
)

charge_marker, = ax.plot([],[], 'wo', markersize=8)
trace_line, = ax.plot([],[], 'w--', linewidth=1)
tx=[]; ty=[]


def update(i):
    t = times[i]
    Ex,Ey,Ez = field_engine.calculate_E(t,X,Y,Z,'Total',plane=True)

    Emag = np.sqrt(Ex**2 + Ey**2 + Ez**2)
    Emag /= Emag.max() + 1e-30
    im.set_data(Emag)

    Exq = Ex[::ds,::ds]
    Eyq = Ey[::ds,::ds]

    vmax = np.percentile(np.sqrt(Exq**2 + Eyq**2), 95)
    Exq = np.clip(Exq, -vmax, vmax)
    Eyq = np.clip(Eyq, -vmax, vmax)

    Q.set_UVC(Exq, Eyq)

    cx = charge.xpos(t)
    cy = charge.ypos(t)
    charge_marker.set_data([cx],[cy])

    tx.append(cx); ty.append(cy)
    trace_line.set_data(tx,ty)
    return [im,Q,charge_marker,trace_line]


ani = animation.FuncAnimation(fig, update, frames=FRAMES, interval=50, blit=False)
ani.save("orbiting_charge.gif", writer="pillow", fps=20)
plt.show()
