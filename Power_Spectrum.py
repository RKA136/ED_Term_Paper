import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import scipy.constants as const

eps = const.epsilon_0
pi  = const.pi
e   = const.e
c   = const.c

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
        return np.sqrt((X - self.xpos(tr))**2 + (Y - self.ypos(tr))**2 + (Z - self.zpos(tr))**2) - c*(t - tr)

class LinearDeceleratingCharge(Charge):
    def __init__(self, pos_charge=True, initial_speed=0.01*c, deceleration=None, stop_t=None):
        super().__init__(pos_charge)
        self.initial_speed = initial_speed
        if deceleration is not None:
            self.deceleration = deceleration
            self.stop_t = initial_speed / deceleration
        elif stop_t is not None:
            self.stop_t = stop_t
            self.deceleration = initial_speed / stop_t
        else:
            self.stop_t = 1e-10
            self.deceleration = initial_speed / self.stop_t

    def xpos(self, t):
        t = np.array(t)
        x = np.zeros_like(t)
        mask1 = (t > 0) & (t <= self.stop_t)
        x[mask1] = self.initial_speed * t[mask1] - 0.5 * self.deceleration * t[mask1]**2
        mask2 = (t > self.stop_t)
        x[mask2] = self.initial_speed * self.stop_t - 0.5 * self.deceleration * self.stop_t**2
        return x

    def ypos(self, t): return 0 * np.array(t)
    def zpos(self, t): return 0 * np.array(t)

    def xvel(self, t):
        t = np.array(t)
        v = np.zeros_like(t)
        mask1 = (t > 0) & (t <= self.stop_t)
        v[mask1] = self.initial_speed - self.deceleration * t[mask1]
        mask2 = (t > self.stop_t)
        v[mask2] = 0.0
        return v

    def yvel(self, t): return 0 * np.array(t)
    def zvel(self, t): return 0 * np.array(t)

    def xacc(self, t):
        t = np.array(t)
        a = np.zeros_like(t)
        mask = (t > 0) & (t <= self.stop_t)
        a[mask] = -self.deceleration
        return a

    def yacc(self, t): return 0 * np.array(t)
    def zacc(self, t): return 0 * np.array(t)


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

        ux = c * rx / r - vx
        uy = c * ry / r - vy
        uz = c * rz / r - vz

        r_dot_u = rx*ux + ry*uy + rz*uz
        r_dot_a = rx*ax + ry*ay + rz*az
        vmag2 = vx**2 + vy**2 + vz**2

        const0 = (e if charge.pos_charge else -e) / (4*pi*eps) * r / (r_dot_u**3)

        Ex_v = const0 * (c**2 - vmag2) * ux
        Ey_v = const0 * (c**2 - vmag2) * uy
        Ez_v = const0 * (c**2 - vmag2) * uz

        Ex_a = const0 * (r_dot_a*ux - r_dot_u*ax)
        Ey_a = const0 * (r_dot_a*uy - r_dot_u*ay)
        Ez_a = const0 * (r_dot_a*uz - r_dot_u*az)

        return Ex_v + Ex_a, Ey_v + Ey_a, Ez_v + Ez_a

    def calculate_E(self, t, X, Y, Z):
        t_arr = np.ones_like(X) * t
        Ex = np.zeros_like(X)
        Ey = np.zeros_like(X)
        Ez = np.zeros_like(X)
        for ch in self.charges:
            tr = optimize.newton(ch.retarded_time, t_arr, args=(t_arr, X, Y, Z), tol=self.h, maxiter=100)
            ex,ey,ez = self._calculate_individual_E(ch, tr, X, Y, Z)
            Ex += ex; Ey += ey; Ez += ez
        return Ex, Ey, Ez


v0 = 0.01 * c           
stop_t = 5e-11          
dec = v0 / stop_t       
charge = LinearDeceleratingCharge(pos_charge=True, initial_speed=v0, deceleration=dec, stop_t=stop_t)
field = MovingChargesField(charge)

Lscale = 6e-9
Robs = 1e3 * Lscale   
Xobs = 0.0
Yobs = Robs
Zobs = 0.0

N = 4096                    
Ttotal = 4 * stop_t        
times = np.linspace(0.0, Ttotal, N, endpoint=False)
dt = times[1] - times[0]


Ey_t = np.zeros(N, dtype=float)
for i, t in enumerate(times):
    Ex, Ey, Ez = field.calculate_E(t,
                                   np.array([[[Xobs]]]),
                                   np.array([[[Yobs]]]),
                                   np.array([[[Zobs]]]))
    Ey_t[i] = Ey[0,0,0]


Ey_t = Ey_t - np.mean(Ey_t)


Ew = np.fft.fft(Ey_t)
freq = np.fft.fftfreq(N, dt)
omega = 2 * np.pi * freq

P_num = np.abs(Ew)**2
mask = omega > 0
omega_pos = omega[mask]
Pnum_pos = P_num[mask]
Pnum_pos /= Pnum_pos.max()


plt.figure(figsize=(8,5))
plt.loglog(omega_pos, Pnum_pos, lw=1.6)
plt.xlabel(r'$\omega$ (rad s$^{-1}$)')
plt.ylabel('Normalized power')
plt.grid(which='both', ls=':')
plt.title('Power spectrum Bremsstrahlung (Not to Scale)')
plt.tight_layout()
plt.savefig("Bremsstrahlung_spectral_shape.png", dpi=300)
plt.show()


