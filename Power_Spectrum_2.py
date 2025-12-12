import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import scipy.constants as const

eps = const.epsilon_0
pi  = const.pi
e   = const.e
c   = const.c
m_e = const.m_e

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
        return np.sqrt((X-self.xpos(tr))**2 + (Y-self.ypos(tr))**2 + (Z-self.zpos(tr))**2) - c*(t - tr)

class OrbitingCharge(Charge):
    def __init__(self, radius=2e-9, max_speed=0.99*c, phase=0.0, pos_charge=True):
        super().__init__(pos_charge)
        self.R = radius
        self.w = max_speed / radius
        self.phase = phase
    def xpos(self,t): return self.R * np.cos(self.w*t + self.phase)
    def ypos(self,t): return self.R * np.sin(self.w*t + self.phase)
    def zpos(self,t): return 0*t
    def xvel(self,t): return -self.R*self.w*np.sin(self.w*t + self.phase)
    def yvel(self,t): return  self.R*self.w*np.cos(self.w*t + self.phase)
    def zvel(self,t): return 0*t
    def xacc(self,t): return -self.R*self.w**2*np.cos(self.w*t + self.phase)
    def yacc(self,t): return -self.R*self.w**2*np.sin(self.w*t + self.phase)
    def zacc(self,t): return 0*t

class MovingChargesField:
    def __init__(self, charges, h=1e-12):
        try:
            len(charges)
        except TypeError:
            charges = [charges]
        self.charges = charges
        self.h = h
    def _calculate_individual_E(self, charge, tr, X, Y, Z):
        rx = X - charge.xpos(tr)
        ry = Y - charge.ypos(tr)
        rz = Z - charge.zpos(tr)
        r = np.sqrt(rx**2 + ry**2 + rz**2)
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
        const0 = (e if charge.pos_charge else -e)/(4*pi*eps) * r/(r_dot_u**3)
        Ex_v = const0*(c**2 - vmag2)*ux
        Ey_v = const0*(c**2 - vmag2)*uy
        Ez_v = const0*(c**2 - vmag2)*uz
        Ex_a = const0*(r_dot_a*ux - r_dot_u*ax)
        Ey_a = const0*(r_dot_a*uy - r_dot_u*ay)
        Ez_a = const0*(r_dot_a*uz - r_dot_u*az)
        return Ex_v + Ex_a, Ey_v + Ey_a, Ez_v + Ez_a
    def calculate_E(self, t, X, Y, Z):
        tarr = np.ones_like(X) * t
        Ex = np.zeros_like(X)
        Ey = np.zeros_like(X)
        Ez = np.zeros_like(X)
        for ch in self.charges:
            tr = optimize.newton(ch.retarded_time, tarr, args=(tarr,X,Y,Z), tol=self.h, maxiter=200)
            ex,ey,ez = self._calculate_individual_E(ch, tr, X, Y, Z)
            Ex += ex; Ey += ey; Ez += ez
        return Ex, Ey, Ez

R = 2e-9
beta = 0.99
v = beta * c
gamma = 1.0 / np.sqrt(1.0 - beta**2)
charge = OrbitingCharge(radius=R, max_speed=beta*c, phase=0.0, pos_charge=True)
field = MovingChargesField(charge)

omega_0 = v / R
T0 = 2*np.pi/omega_0
omega_c = 1.5 * gamma**3 * c / R   

print("gamma =", gamma)
print("omega_c / omega_0 =", omega_c/omega_0)

Robs = 200.0 * R     
nx, ny, nz = 1.0, 0.0, 0.0
Xobs = nx * Robs
Yobs = ny * Robs
Zobs = nz * Robs
N_periods = 40
N_per_period = 256
Nt = int(N_periods * N_per_period)
dt = T0 / N_per_period
times = np.linspace(0.0, N_periods*T0, Nt, endpoint=False)

Ey_t = np.zeros(Nt, dtype=np.float64)
Ez_t = np.zeros(Nt, dtype=np.float64)

print("Sampling E(t) at observer, Nt =", Nt)
for i, t in enumerate(times):
    Ex, Ey, Ez = field.calculate_E(t,
                                   np.array([[[Xobs]]]),
                                   np.array([[[Yobs]]]),
                                   np.array([[[Zobs]]]))
    Ey_t[i] = Ey[0,0,0]
    Ez_t[i] = Ez[0,0,0]

Ey_t -= np.mean(Ey_t)
Ez_t -= np.mean(Ez_t)

E_t = Ey_t + 1j * Ez_t

Ew = np.fft.fft(E_t)
freqs = np.fft.fftfreq(Nt, dt)
omega = 2*np.pi * freqs

P_num = np.abs(Ew)**2
mask = omega > 0
omega_pos = omega[mask]
Pnum_pos = P_num[mask]
Pnum_pos = Pnum_pos / np.nanmax(Pnum_pos)

plt.figure(figsize=(8,5))
plt.loglog(omega_pos/omega_c, Pnum_pos, lw=1.5)
plt.xlabel(r'$\omega/\omega_c$')
plt.ylabel('Normalized power (arb. units)')
plt.grid(which='both', ls=':')
plt.xlim(1e-3, 1e2)
plt.ylim(1e-6, 1.2)
plt.title('Synchrotron spectral shape (Not to Scale)')
plt.tight_layout()
plt.savefig("Synchrotron_spectral_shape.png", dpi=300)
plt.show()