import numpy as np


def f(x, occ, c):
    return (np.sum(x.T/(x.T+occ.T), axis=0)-c.T).T


def fprime(x, occ, c=0):
    return np.sum(occ.T/(x.T+occ.T)**2, axis=0).T


def fprime2(x, occ, c=0):
    return -2*np.sum(occ.T/(x.T+occ.T)**3, axis=0).T


def get_factor(occ, c, tol=1e-8):
    x = np.zeros_like(c)
    while 1:
        ff = f(x, occ, c)
        if np.linalg.norm(ff)/np.prod(c.shape) < tol:
            return 1/x
        ffprime = fprime(x, occ, c)
        x -= ff/ffprime


dipole_tensor = np.array([
    3.400301119999999955e+00, 3.400301119999999955e+00, 8.030965309999999135e+00
])
dipole_tensor = np.stack([np.roll(dipole_tensor, ii+1) for ii in range(3)], axis=0)


class Carbon:
    def __init__(
        self,
        medium,
        L=(400, 40, 40),
        N=(100, 10, 10),
        dipole_tensor=dipole_tensor,
        a_0=2.85,
        initial_strain=None,
        temperature=600,
        diffusion_coefficient=1,
        force_constant=0,
    ):
        self.L = L
        self.N = N
        self.dipole_tensor = dipole_tensor
        self._k_mesh = None
        self.medium = medium
        self.c = np.zeros(self.N)
        if initial_strain is None:
            initial_strain = np.zeros((3, 3))
            initial_strain[2,2] = 0.1
        self.current_strain = np.einsum('...,ij->...ij', np.ones_like(self.c), initial_strain)
        self.a_0 = a_0
        self.temperature = temperature
        self.D = diffusion_coefficient
        self.force_constant = force_constant
        self._G_k = None
        self._G_self = None
        self.fermi = False
        self._c_density = None

    def initialize(self):
        self._k_mesh = None
        self._G_k = None
        self._G_self = None
        self._c_density = None

    @property
    def c_density(self):
        if self._c_density is None:
            self._c_density = 2/self.a_0**3*self.measure
        return self._c_density

    @property
    def measure(self):
        return np.prod(self.L)/np.prod(self.N)

    @property
    def spacing(self):
        return np.array(self.L)/np.array(self.N)

    @property
    def L(self):
        return self._L

    @L.setter
    def L(self, ll):
        self._L = ll
        self.initialize()

    @property
    def N(self):
        return self._N

    @N.setter
    def N(self, nn):
        self._N = nn
        self.initialize()

    @property
    def dv(self):
        return np.prod(self.L)/np.prod(self.N)

    @property
    def k_mesh(self):
        if self._k_mesh is None:
            k_lin = [
                np.roll(np.pi/l*np.linspace(-n, n, n, endpoint=False), n//2)
                for n, l in zip(self.N, self.L)
            ]
            self._k_mesh = np.einsum('nxyz->xyzn', np.meshgrid(*k_lin, indexing='ij'))
            self._k_mesh[np.linalg.norm(self._k_mesh/np.pi*np.array(self.L)/np.array(self.N), axis=-1) > 1] = 0
        return self._k_mesh

    @property
    def filter_(self):
        return np.linalg.norm(self.k_mesh, axis=-1) > 0

    @property
    def G_k(self):
        if self._G_k is None:
            self._G_k = np.zeros(self.N+(3, 3))
            self._G_k[self.filter_] = self.medium.get_greens_function(self.k_mesh[self.filter_], fourier=True)
            if self.force_constant > 0:
                self._G_k = np.einsum(
                    '...ij,...->...ij',
                    self._G_k,
                    np.exp(-self.kBT/self.force_constant*np.sum(self.k_mesh**2, axis=-1))
                )
        return self._G_k

    @property
    def G_self(self):
        if self._G_self is None:
            self._G_self = np.einsum(
                'xyzj,xyzl,xyzik->ijkl', self.k_mesh, self.k_mesh, self.G_k, optimize=True
            )
            self._G_self = 0.5*(self._G_self+np.einsum('ijkl->jikl', self._G_self))/np.prod(self.N)
        return self._G_self

    @property
    def self_strain(self):
        return np.einsum('...k,ijkk->...ij', self.P, self.G_self)

    @property
    def strain_zero(self):
        return self.medium.compliance_matrix[:3,:3].dot(
            self.dipole_tensor.dot(np.mean(self.c_partial, axis=(0,1,2)))
        )*6/self.a_0**3*np.eye(3)

    @property
    def c_partial(self):
        E = np.einsum(
            'ij,...jj->...i', self.dipole_tensor, self.current_strain
        )
        if self.fermi:
            occ = np.exp(-E/self.kBT/self.c_density)
            A = get_factor(occ, self.c)
            return 1/(1+A.T*occ.T).T
        occ = np.exp(E/self.kBT/self.c_density)
        return np.einsum('...,...i,...->...i', self.c, occ, 1/np.sum(occ, axis=-1))

    @property
    def P(self):
        return np.einsum('...i,ik->...k', self.c_partial, self.dipole_tensor)*self.c_density

    @property
    def _P_k(self):
        return np.fft.fftn(self.P, axes=(0, 1, 2))

    @property
    def strain(self):
        strain_zero = self.strain_zero.copy()
        self.current_strain = np.einsum(
            'xyzj,xyzk,xyzik,xyzk->xyzij', self.k_mesh, self.k_mesh, self.G_k, self._P_k, optimize=True
        )
        self.current_strain = np.real(np.fft.ifftn(self.current_strain, axes=(0, 1, 2)))
        self.current_strain -= self.self_strain
        self.current_strain /= self.measure
        self.current_strain += strain_zero
        return self.current_strain

    @property
    def nabla_strain(self):
        return -np.real(np.fft.ifftn(np.einsum(
            '...l,...j,...k,...ik,...k->...ijl',
            self.k_mesh, self.k_mesh, self.k_mesh, self.G_k, self._P_k, optimize=True
        )*1j, axes=(0, 1, 2)))/np.array(self.N)

    @property
    def laplace_strain(self):
        return -np.real(np.fft.ifftn(np.einsum(
            '...l,...l,l,...j,...k,...ik,...k->...ij',
            self.k_mesh, self.k_mesh, 1/np.array(self.N)**2, self.k_mesh, self.k_mesh, self.G_k, self._P_k,
            optimize=True
        ), axes=(0, 1, 2)))

    @property
    def displacement(self):
        u_k = -np.einsum('xyzk,xyzik,xyzk->xyzi', self.k_mesh, self.G_k, self._P_k, optimize=True)*1j
        return np.real(np.fft.ifftn(u_k, axes=(0, 1, 2)))

    @property
    def kBT(self):
        return 8.617e-5*self.temperature

    @property
    def nabla_c(self):
        return np.einsum('i...->...i', [
            (np.roll(self.c, -1, axis=ii)-np.roll(self.c, 1, axis=ii))/ss/2
            for ii, ss in enumerate(self.spacing)
        ])

    @property
    def laplace_c(self):
        return np.sum([
            (np.sum([np.roll(self.c, jj, axis=ii) for jj in [-1, 1]], axis=0)-2*self.c)/ss**2
            for ii, ss in enumerate(self.spacing)
        ], axis=0)

    @property
    def dUdt(self):
        de = np.einsum('...,...iik,in->...k', 1-2*self.c, self.nabla_strain, self.dipole_tensor)
        dde = np.einsum('...,...,...ii,in->...', self.c, 1-self.c, self.laplace_strain, self.dipole_tensor)
        return -self.D/self.kBT*(np.sum(de*self.nabla_c, axis=-1)+dde)

    @property
    def dSdt(self):
        return self.c_density*self.D*self.laplace_c

    @property
    def dcdt(self):
        dcdt = self.dUdt+self.dSdt
        return dcdt-np.mean(dcdt)

    @property
    def free_energy(self):
        return self.internal_energy-self.temperature*self.entropy

    @property
    def entropy(self):
        c = self.c_partial
        return -8.617e-5*self.c_density*np.sum(c*np.log(c)+(1-c)*np.log(1-c))

    @property
    def internal_energy(self):
        return -np.einsum('xyzi,xyzii->', self.P, self.strain)

    @property
    def order_parameter(self):
        inner = 3/2*np.sum(self.c_partial**2, axis=-1)/self.c**2-1/2
        inner[inner<0] = 0
        return np.sqrt(inner)

    @property
    def total_order_parameter(self):
        return np.sqrt(3/2*np.sum(np.mean(self.c_partial, axis=(0,1,2))**2)/np.mean(self.c)**2-1/2)