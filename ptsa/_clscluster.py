
    def to_clsT(self, unit_wavelength="nm"):
        """
        Convert T-matrix to an object of class clsT (Xavi's code)

        Returns:
            object of class clsCluster
        """

        from clsT import clsT

        if self.positions.shape[0] > 1:
            warnings.warn(
                """
                    This T-matrix can not be converted into an object of class clsT as
                    it contains more than one scatterer. An object of class clsCluster
                    will be returned instead
                """
            )
            return self.to_clsCluster()
        if self.helicity:
            basis = "helicity"
        else:
            basis = "electric-magnetic"
        n_max = int(np.max(self.modes[0]))
        modes = (
            *np.array(
                [
                    [n, m, p]
                    for p in range(1, -1, -1)
                    for n in range(1, n_max + 1)
                    for m in range(-n, n + 1)
                ]
            ).T,
        )
        mat = misc.pickmodes(self.modes, modes)
        tmatrix = mat.T @ self.t @ mat
        return clsT(
            tmatrix,
            basis,
            wavelength=2 * np.pi / self.k0,
            refr_index=np.sqrt(self.epsilon * self.mu),
            wavelength_units=unit_wavelength,
        )

    def to_clsCluster(self):
        """
        Convert T-matrix to an object of class clsCluster (Xavi's code)

        Returns:
            object of class clsCluster
        """

        from clsCluster import clsCluster
        from clsT import clsT

        if self.helicity:
            basis = "helicity"
        else:
            basis = "electric-magnetic"
        number_particles = self.positions.shape[0]
        max_n = np.zeros((number_particles,), int)
        for i_particle in range(number_particles):
            max_n[i_particle] = np.max(self.l[self.pidx == i_particle])
        modes = (
            *np.array(
                [
                    [np, n, m, p]
                    for np in range(number_particles)
                    for p in range(1, -1, -1)
                    for n in range(1, max_n[np] + 1)
                    for m in range(-n, n + 1)
                ]
            ).T,
        )
        mat = misc.pickmodes(self.fullmodes, modes)
        tmatrix = mat.T @ self.t @ mat
        ts = []
        for i_particle in range(number_particles):
            entries = modes[0] == i_particle
            ts.append(
                clsT(
                    tmatrix[entries, :][:, entries],
                    basis,
                    wavelength=2 * np.pi / self.k0,
                    refr_index=np.sqrt(self.epsilon * self.mu),
                )
            )
        cluster = clsCluster(self.positions, ts)
        return cluster

    def effective_optical_parameters(self, concentration, units="nm"):
        """
        Computes the effective optical parameters of a medium with immersed scatteres
        given by their T-matrix. The effective parameters are obtained using
        Clausius-Mosotti homegenization equations.
        Authors: This code is an adaptation made by Xavi from an original script created
        by Benedikt Zerulla.

        Args:
            self (_tmatrix): object of class _tmatrix
            concentration (float): concentration of scatterers per cubic meter of host
                medium.
            units (string): units in which k0 is given in the _tmatrix object. By
                default it is assumed to be in nm^{-1}.

        Returns:
            tuple: tuple containing three float numbers corresponding to the relative
                permittivity, relative permeability and relative chirality parameter.
                (eff_eps, eff_mu, eff_kappa)
        """

        eps0 = 8.8541878128 * 1e-12
        mu0 = 1.2566370621 * 1e-6
        epsilon = self.epsilon * eps0
        mu = self.mu * mu0
        c = 1 / np.sqrt(eps0 * self.epsilon * mu0 * self.mu)
        Z = np.sqrt(mu / epsilon)
        if self.helicity:
            self.changebasis()
        modes = (6 * [1], 2 * [-1, 0, 1], 3 * [1] + 3 * [0])
        mat = misc.pickmodes(self.modes, modes)
        T = mat.T @ self.t @ mat
        k0 = io._convert_to_k0(np.copy(self.k0), "k0", units + r"^{-1}", r"nm^{-1}")
        k_mod = 1e9 * k0 * np.sqrt(self.epsilon * self.mu)
        scaling = (
            -6j * np.pi / (c * Z * k_mod ** 3) * np.array([1, 1j * Z, -1j * c, c * Z])
        )
        vector = np.array(
            [
                [np.sqrt(0.5), 1j * np.sqrt(0.5), 0],
                [0, 0, 1],
                [-np.sqrt(0.5), 1j * np.sqrt(0.5), 0],
            ]
        )
        alpha_mean = []
        size_t = T.shape[0] // 2
        index = np.array(
            [
                [0, size_t, 0, size_t],
                [0, size_t, size_t, 2 * size_t],
                [size_t, 2 * size_t, 0, size_t],
                [size_t, 2 * size_t, size_t, 2 * size_t],
            ]
        )
        for i_pol in range(4):
            alphaSI = (
                scaling[i_pol]
                * T[
                    index[i_pol, 0] : index[i_pol, 1], index[i_pol, 2] : index[i_pol, 3]
                ]
            )
            alpha = vector.T.conjugate() @ (alphaSI @ vector)
            alpha_mean.append(np.sum(np.diag(alpha)) / 3)
        delta_alpha = (
            alpha_mean[0] * alpha_mean[3] * mu - alpha_mean[1] * alpha_mean[2] * mu
        )
        D = (
            1
            - concentration * alpha_mean[0] / (3 * epsilon)
            - concentration * alpha_mean[3] / 3
            + concentration ** 2 * delta_alpha / (9 * mu * epsilon)
        )
        epsilon_eff = (
            epsilon
            + (
                concentration * alpha_mean[0]
                - concentration ** 2 * delta_alpha / (3 * mu)
            )
            / D
        ) / eps0
        mu_eff = (
            mu
            + (
                concentration * alpha_mean[3] * mu
                - concentration ** 2 * delta_alpha / (3 * epsilon)
            )
            / D
        ) / mu0
        kappa_eff = -1j * concentration * alpha_mean[1] / np.sqrt(mu0 * eps0) / D
        return epsilon_eff, mu_eff, kappa_eff
