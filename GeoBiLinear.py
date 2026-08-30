from kingdon import Algebra
import torch
import torch.nn as nn


def build_cayley_table_kingdon():
    """
    Build geometric product Cayley table for G(3,0,1) using kingdon.
    Returns a (16, 16, 16) tensor.
    """
    alg = Algebra(3, 0, 1)

    # get all 16 basis blades as kingdon multivectors
    basis = [alg.multivector({i: 1.0}) for i in range(16)]

    table = torch.zeros(16, 16, 16)
    for i in range(16):
        for j in range(16):
            # compute geometric product of basis blade i and j
            product = basis[i] * basis[j]
            # extract coefficients into the table
            for k, val in product.items():
                table[i, j, k] = float(val)

    return table


def build_outer_product_table_kingdon():
    """
    Build outer (wedge) product table for G(3,0,1) using kingdon.
    Returns a (16, 16, 16) tensor.
    """
    alg = Algebra(3, 0, 1)
    basis = [alg.multivector({i: 1.0}) for i in range(16)]

    table = torch.zeros(16, 16, 16)
    for i in range(16):
        for j in range(16):
            product = basis[i] ^ basis[j]  # ^ is wedge in kingdon
            for k, val in product.items():
                table[i, j, k] = float(val)

    return table


# precompute once at module level
print("Building Cayley tables...")
CAYLEY_TABLE = build_cayley_table_kingdon()
OUTER_PRODUCT_TABLE = build_outer_product_table_kingdon()
print("Done.")


def geometric_product(x, y):
    """
    x, y   : (B, N, C, 16)
    returns : (B, N, C, 16)
    """
    cayley = CAYLEY_TABLE.to(x.device)
    return torch.einsum('...i, ...j, ijk -> ...k', x, y, cayley)


def outer_product(x, y):
    """
    x, y   : (B, N, C, 16)
    returns : (B, N, C, 16)
    """
    table = OUTER_PRODUCT_TABLE.to(x.device)
    return torch.einsum('...i, ...j, ijk -> ...k', x, y, table)


def dual(x):
    """
    Dual in G(3,0,1) — reverses grades.
    x: (B, N, C, 16)
    returns: (B, N, C, 16)
    """
    out = torch.zeros_like(x)
    out[..., 15] = x[..., 0]
    out[..., 14] = x[..., 1]
    out[..., 13] = x[..., 2]
    out[..., 12] = x[..., 3]
    out[..., 11] = x[..., 4]
    out[..., 10] = x[..., 5]
    out[..., 9] = x[..., 6]
    out[..., 8] = x[..., 7]
    out[..., 7] = x[..., 8]
    out[..., 6] = x[..., 9]
    out[..., 5] = x[..., 10]
    out[..., 4] = x[..., 11]
    out[..., 3] = x[..., 12]
    out[..., 2] = x[..., 13]
    out[..., 1] = x[..., 14]
    out[..., 0] = x[..., 15]
    return out


def equi_join(x, y, z):
    """
    Equivariant join: z₀₁₂₃ · (x* ∧ y*)*

    x, y, z : (B, N, C, 16)
    returns  : (B, N, C, 16)
    """
    # pseudoscalar component of z → (B, N, C, 1)
    z_pseudo = z[..., 15:16]

    # dual → outer product → dual
    result = z_pseudo * dual(outer_product(dual(x), dual(y)))

    return result


class GeoBilinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('cayley', CAYLEY_TABLE)
        self.register_buffer('outer', OUTER_PRODUCT_TABLE)

        # precompute sparse indices at init — do this ONCE
        cayley_idx, cayley_val = self._sparse_entries(CAYLEY_TABLE)
        outer_idx, outer_val = self._sparse_entries(OUTER_PRODUCT_TABLE)

        self.register_buffer('cayley_idx', cayley_idx)
        self.register_buffer('cayley_val', cayley_val)
        self.register_buffer('outer_idx', outer_idx)
        self.register_buffer('outer_val', outer_val)

    def _sparse_entries(self, table):
        """Extract nonzero (i, j, k, value) entries from a 16x16x16 table."""
        idx = torch.nonzero(table, as_tuple=False)  # (nnz, 3)
        val = table[idx[:, 0], idx[:, 1], idx[:, 2]]  # (nnz,)
        return idx, val

    def _sparse_product(self, x, y, idx, val):
        """
        Sparse geometric product.
        x, y : (..., 16)
        Only computes nonzero blade multiplications.
        """
        i, j, k = idx[:, 0], idx[:, 1], idx[:, 2]

        # for each nonzero entry: contribution = val * x[...,i] * y[...,j]
        contrib = val * x[..., i] * y[..., j]  # (..., nnz)

        out = torch.zeros_like(x)
        # scatter contributions into output blade k
        out.scatter_add_(-1, k.expand_as(contrib), contrib)
        return out

    def forward(self, x, y, z):
        gp = self._sparse_product(x, y, self.cayley_idx, self.cayley_val)
        ej = self._equi_join(x, y, z)
        return torch.cat([gp, ej], dim=-2)

    def _equi_join(self, x, y, z):
        z_pseudo = z[..., 15:16]
        wx = self._sparse_product(
            dual(x), dual(y),
            self.outer_idx, self.outer_val
        )
        return z_pseudo * dual(wx)

