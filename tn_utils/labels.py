import opt_einsum as oe


class Labels:
    """Build labels for MPS contraction einsum strings."""

    def __init__(self, n_sites: int) -> None:
        """The constructor.

        Parameters
        ----------
        n_sites: int
            Number of MPS sites
        """
        self.phys_labels = [oe.get_symbol(k) for k in range(n_sites)]
        self.tensor_labels = [
            "{}{}{}".format(
                oe.get_symbol(k), self.phys_labels[k - n_sites], oe.get_symbol(k + 1)
            )
            for k in range(n_sites, 2 * n_sites)
        ]

    def psi_einsum_str(self) -> str:
        """Einsum string for psi(x)

        Returns
        -------
        The einsum string
        """
        return ",".join(self.tensor_labels + self.phys_labels)

    def psic_einsum_str(self, cbond: tuple[int, int]) -> str:
        """Einsum string for psi(x) with a contracted bond.

        Parameters
        ----------
        cbond: tuple[int, int]
            (left_index, right_index), the MPS sites that are contracted.

        Returns
        -------
        The einsum string
        """
        lidx, ridx = cbond
        clabel = self.tensor_labels[lidx][:2] + self.tensor_labels[ridx][1:]
        return ",".join(
            self.tensor_labels[:lidx]
            + [clabel]
            + self.tensor_labels[ridx + 1 :]
            + self.phys_labels
        )
