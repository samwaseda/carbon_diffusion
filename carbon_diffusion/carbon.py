import numpy as np
from pyiron_continuum.elasticity.linear_elasticity import LinearElasticity


def f(x, occ, c):
    return (np.sum(x.T / (x.T + occ.T), axis=0) - c.T).T


def fprime(x, occ, c=0):
    return np.sum(occ.T / (x.T + occ.T)**2, axis=0).T


def fprime2(x, occ, c=0):
    return -2 * np.sum(occ.T / (x.T + occ.T)**3, axis=0).T


def get_factor(occ, c, tol=1e-8):
    x = np.zeros_like(c)
    while 1:
        ff = f(x, occ, c)
        if np.linalg.norm(ff) / np.prod(c.shape) < tol:
            return 1 / x
        ffprime = fprime(x, occ, c)
        x -= ff / ffprime


elastic_tensor = np.array([
    1.518305767738742151e+00, 9.055162512455107171e-01, 7.245888668410520594e-01
])
dipole_tensor = np.array([
    3.400301119999999955e+00, 3.400301119999999955e+00, 8.030965309999999135e+00
])
dipole_tensor = np.stack([np.roll(dipole_tensor, ii + 1) for ii in range(3)], axis=0)


class Carbon:
    def __init__(
        self,
        medium=LinearElasticity(elastic_tensor),
        L=(400, 40, 40),
        N=(100, 10, 10),
        dipole_tensor=dipole_tensor,
        a_0=2.85,
        initial_strain=None,
        temperature=600,
        diffusion_coefficient=1,
        fill_value=0.01,
        force_constant=0,
        chemical_repulsion=0,
    ):
        self.L = L
        self.N = N
        self.dipole_tensor = dipole_tensor
        self._k_mesh = None
        self.medium = medium
        self.c = np.ones(self.N) * fill_value
        if initial_strain is None:
            initial_strain = np.zeros((3, 3))
            initial_strain[2, 2] = 0.1
        self.current_strain = np.einsum('...,ij->...ij', np.ones_like(self.c), initial_strain)
        self.a_0 = a_0
        self.temperature = temperature
        self.D = diffusion_coefficient
        self.force_constant = force_constant
        self._G_k = None
        self._G_self = None
        self.fermi = False
        self._c_density = None
        self._c_partial = None
        self.chemical_repulsion = chemical_repulsion

    def initialize(self):
        self._k_mesh = None
        self._G_k = None
        self._G_self = None
        self._c_density = None

    @property
    def c_density(self):
        if self._c_density is None:
            self._c_density = 2 / self.a_0**3 * self.measure
        return self._c_density

    @property
    def measure(self):
        return np.prod(self.L) / np.prod(self.N)

    @property
    def spacing(self):
        return np.array(self.L) / np.array(self.N)

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
        return np.prod(self.L) / np.prod(self.N)

    @property
    def mesh(self):
        lin = [
            np.linspace(0, l, n, endpoint=False)
            for n, l in zip(self.N, self.L)
        ]
        return np.einsum('nxyz->xyzn', np.meshgrid(*lin, indexing='ij'))

    @property
    def k_mesh(self):
        if self._k_mesh is None:
            k_lin = [
                np.roll(np.pi / l * np.linspace(-n, n, n, endpoint=False), n // 2)
                for n, l in zip(self.N, self.L)
            ]
            self._k_mesh = np.einsum('nxyz->xyzn', np.meshgrid(*k_lin, indexing='ij'))
            self._k_mesh[np.linalg.norm(self._k_mesh / np.pi * np.array(self.L) / np.array(self.N), axis=-1) > 1] = 0
        return self._k_mesh

    @property
    def filter_(self):
        return np.linalg.norm(self.k_mesh, axis=-1) > 0

    @property
    def G_k(self):
        if self._G_k is None:
            self._G_k = np.zeros(self.N + (3, 3))
            self._G_k[self.filter_] = self.medium.get_greens_function(self.k_mesh[self.filter_], fourier=True)
            if self.force_constant > 0:
                self._G_k = np.einsum(
                    '...ij,...->...ij',
                    self._G_k,
                    np.exp(-self.kBT / self.force_constant * np.sum(self.k_mesh**2, axis=-1))
                )
        return self._G_k

    @property
    def G_self(self):
        if self._G_self is None:
            self._G_self = np.einsum(
                'xyzj,xyzl,xyzik->ijkl', self.k_mesh, self.k_mesh, self.G_k, optimize=True
            )
            self._G_self = 0.5 * (self._G_self + np.einsum('ijkl->jikl', self._G_self)) / np.prod(self.N)
        return self._G_self

    @property
    def self_strain(self):
        return np.einsum('...k,ijkk->...ij', self.P, self.G_self)

    @property
    def strain_zero(self):
        return self.medium.compliance_matrix[:3, :3].dot(
            self.dipole_tensor.dot(np.mean(self.c_partial, axis=(0, 1, 2)))
        ) * 6 / self.a_0**3 * np.eye(3)

    @property
    def c_partial(self):
        if self._c_partial is None:
            E = np.einsum(
                'ij,...jj->...i', self.dipole_tensor, self.current_strain
            )
            occ = np.exp(-E / self.kBT / self.c_density)
            if self.fermi:
                A = get_factor(occ, self.c)
                self._c_partial = 1 / (1 + A.T * occ.T).T
            else:
                self._c_partial = np.einsum(
                    '...,...i,...->...i', self.c, 1 / occ, 1 / np.sum(1 / occ, axis=-1)
                )
        return self._c_partial

    @property
    def P(self):
        return np.einsum('...i,ik->...k', self.c_partial, self.dipole_tensor) * self.c_density

    @property
    def _P_eff(self):
        return np.einsum(
            '...i,ik,...->...k', self.c_partial, self.dipole_tensor, 1 / self.c
        ) * self.c_density

    @property
    def _P_k(self):
        return np.fft.fftn(self.P, axes=(0, 1, 2))

    @property
    def strain(self):
        strain_zero = self.strain_zero.copy()
        self.current_strain = np.einsum(
            'xyzj,xyzk,xyzik,xyzk->xyzij', self.k_mesh, self.k_mesh, self.G_k, self._P_k,
            optimize=True
        )
        self.current_strain = np.real(np.fft.ifftn(self.current_strain, axes=(0, 1, 2)))
        self.current_strain -= self.self_strain
        self.current_strain /= self.measure
        self.current_strain += strain_zero
        self._c_partial = None
        return self.current_strain

    @property
    def nabla_strain(self):
        return -np.real(np.fft.ifftn(np.einsum(
            '...l,...j,...k,...ik,...k->...ijl',
            self.k_mesh, self.k_mesh, self.k_mesh, self.G_k, self._P_k, optimize=True
        ) * 1j, axes=(0, 1, 2))) / np.array(self.N)

    @property
    def laplace_strain(self):
        return -np.real(np.fft.ifftn(np.einsum(
            '...l,...l,l,...j,...k,...ik,...k->...ij',
            self.k_mesh, self.k_mesh, 1 / np.array(self.N)**2, self.k_mesh, self.k_mesh, self.G_k,
            self._P_k,
            optimize=True
        ), axes=(0, 1, 2)))

    @property
    def displacement(self):
        u_k = -np.einsum(
            'xyzk,xyzik,xyzk->xyzi', self.k_mesh, self.G_k, self._P_k, optimize=True
        ) * 1j
        return np.real(np.fft.ifftn(u_k, axes=(0, 1, 2)))

    @property
    def kBT(self):
        return 8.617e-5 * self.temperature

    def get_nabla(self, x):
        return np.einsum('i...->...i', [
            (np.roll(x, -1, axis=ii) - np.roll(x, 1, axis=ii)) / ss / 2
            for ii, ss in enumerate(self.spacing)
        ])

    def get_laplace(self, x):
        return np.sum([
            (np.sum([np.roll(x, jj, axis=ii) for jj in [-1, 1]], axis=0) - 2 * x) / ss**2
            for ii, ss in enumerate(self.spacing)
        ], axis=0)

    @property
    def nabla_c(self):
        return self.get_nabla(self.c)

    @property
    def laplace_c(self):
        return self.get_laplace(self.c)

    @property
    def _nabla_u(self):
        return -np.einsum(
            '...jk,...jj->...k', self.get_nabla(self._P_eff), self.strain
        ) - np.einsum('...j,...jjk->...k', self._P_eff, self.nabla_strain)

    @property
    def _laplace_u(self):
        return -np.einsum(
            '...j,...jj->...', self.get_laplace(self._P_eff), self.strain
        ) - 2 * np.einsum(
            '...jk,...jjk->...', self.get_nabla(self._P_eff), self.nabla_strain
        ) - np.einsum('...j,...jj->...', self._P_eff, self.laplace_strain)

    @property
    def dUdt(self):
        dUdt = (1 - 4 * self.c) * np.sum(self.nabla_c * self._nabla_u, axis=-1)
        dUdt += self.c * (1 - 3 * self.c) * self._laplace_u
        if self.chemical_repulsion > 0:
            dUdt += self.c_density * self.chemical_repulsion * (
                (1 - 2 * self.c) * np.sum(self.nabla_c**2, axis=-1) + self.c * (1 - self.c) * self.laplace_c
            )
        return self.D / self.kBT * dUdt

    @property
    def dSdt(self):
        return self.c_density * self.D * self.laplace_c

    @property
    def dcdt(self):
        return self.dUdt + self.dSdt

    @property
    def free_energy(self):
        return self.internal_energy - self.temperature * self.entropy

    @property
    def entropy(self):
        c = self.c_partial
        return -8.617e-5 * self.c_density * np.sum(c * np.log(c) + (1 - c) * np.log(1 - c), axis=-1)

    @property
    def internal_energy(self):
        return -np.einsum('xyzi,xyzii->xyz', self.P, self.strain)

    @property
    def order_parameter(self):
        inner = 3 / 2 * np.sum(self.c_partial**2, axis=-1) / np.sum(self.c_partial, axis=-1)**2 - 1 / 2
        inner[inner < 0] = 0
        return np.sqrt(inner)

    @property
    def total_order_parameter(self):
        f = 3 / 2 * np.sum(np.mean(self.c_partial, axis=(0, 1, 2))**2) / np.mean(self.c)**2
        if f < 0:
            return 0
        else:
            return np.sqrt(f - 1 / 2)
