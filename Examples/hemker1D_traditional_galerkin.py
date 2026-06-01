import os
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
from scipy.linalg import lstsq
from numpy.polynomial.legendre import Legendre


# ============================================================
# Traditional least-squares Galerkin method for 1D Hemker problem
# ============================================================

RESULTS_PATH = "hemker1D_1e-5_traditional_galerkin"
os.makedirs(RESULTS_PATH, exist_ok=True)

myeps = 1.0e-5
boundary_penalty = 1.0e3


def source(x):
    """
    f(x) = eps u'' + x u'
    for the manufactured solution:
        u(x) = cos(pi x) + erf(x / sqrt(2 eps)) / erf(1 / sqrt(2 eps))
    """
    return -myeps * np.pi**2 * np.cos(np.pi * x) - np.pi * x * np.sin(np.pi * x)


def exact(x):
    return np.cos(np.pi * x) + erf(x / np.sqrt(2 * myeps)) / erf(1.0 / np.sqrt(2 * myeps))


def gauss_legendre_rule(nq, a=-1.0, b=1.0):
    """
    Gauss-Legendre quadrature on [a,b].
    """
    xi, wi = np.polynomial.legendre.leggauss(nq)
    x = 0.5 * (b - a) * xi + 0.5 * (a + b)
    w = 0.5 * (b - a) * wi
    return x, w


def build_legendre_basis(max_degree, x):
    """
    Build Legendre basis:
        phi_j(x) = P_j(x), j = 0,...,N

    Since the least-squares formulation contains u'', we use smooth global
    polynomial basis functions instead of piecewise linear FEM basis functions.
    """
    basis = []
    basis_x = []
    basis_xx = []

    for j in range(max_degree + 1):
        P = Legendre.basis(j)
        Px = P.deriv(1)
        Pxx = P.deriv(2)

        basis.append(P(x))
        basis_x.append(Px(x))
        basis_xx.append(Pxx(x))

    return np.array(basis), np.array(basis_x), np.array(basis_xx)


def assemble_system(max_degree, nq=4096):
    """
    Assemble the least-squares Galerkin linear system:

        A_ij = ∫(eps phi_i'' + x phi_i') (eps phi_j'' + x phi_j') dx
               + C * sum_boundary phi_i phi_j

        F_i  = ∫ f (eps phi_i'' + x phi_i') dx
               + C * sum_boundary u_D phi_i

    Boundary is imposed weakly using the same penalty form as the GNN code.
    """
    x, w = gauss_legendre_rule(nq, -1.0, 1.0)

    phi, phix, phixx = build_legendre_basis(max_degree, x)
    n_basis = max_degree + 1

    # L phi_j = eps phi_j'' + x phi_j'
    Lphi = myeps * phixx + x[None, :] * phix

    A = np.zeros((n_basis, n_basis))
    F = np.zeros((n_basis, 1))

    f = source(x)

    # Interior least-squares part
    for i in range(n_basis):
        for j in range(n_basis):
            A[i, j] = np.sum(w * Lphi[i] * Lphi[j])
        F[i, 0] = np.sum(w * f * Lphi[i])

    # Boundary penalty part
    xb = np.array([-1.0, 1.0])
    wb = np.array([1.0, 1.0])

    phi_b, _, _ = build_legendre_basis(max_degree, xb)
    u_b = exact(xb)

    for i in range(n_basis):
        for j in range(n_basis):
            A[i, j] += boundary_penalty * np.sum(wb * phi_b[i] * phi_b[j])
        F[i, 0] += boundary_penalty * np.sum(wb * u_b * phi_b[i])

    return A, F


def evaluate_solution(coeffs, x):
    """
    Evaluate u_N, u_N', u_N''.
    """
    max_degree = len(coeffs) - 1
    phi, phix, phixx = build_legendre_basis(max_degree, x)

    c = coeffs.reshape(-1, 1)

    u = np.sum(c * phi, axis=0)
    ux = np.sum(c * phix, axis=0)
    uxx = np.sum(c * phixx, axis=0)

    return u, ux, uxx


def compute_errors(coeffs, nq_val=4096):
    """
    Compute relative L2 error and relative least-squares energy error.
    """
    x, w = gauss_legendre_rule(nq_val, -1.0, 1.0)

    u_pred, ux_pred, uxx_pred = evaluate_solution(coeffs, x)
    u_true = exact(x)

    l2_err = np.sqrt(np.sum(w * (u_pred - u_true) ** 2))
    l2_ref = np.sqrt(np.sum(w * u_true ** 2))
    rel_l2 = l2_err / l2_ref

    residual = source(x) - (myeps * uxx_pred + x * ux_pred)

    xb = np.array([-1.0, 1.0])
    ub_pred, _, _ = evaluate_solution(coeffs, xb)
    ub_true = exact(xb)

    energy_err = np.sqrt(
        np.sum(w * residual**2)
        + boundary_penalty * np.sum((ub_pred - ub_true) ** 2)
    )

    energy_ref = np.sqrt(
        np.sum(w * source(x) ** 2)
        + boundary_penalty * np.sum(ub_true**2)
    )

    rel_energy = energy_err / energy_ref

    return rel_l2, rel_energy


def solve_for_degree(max_degree, nq_train=4096, nq_val=4096):
    t0 = time.time()

    A, F = assemble_system(max_degree, nq=nq_train)

    cond_A = np.linalg.cond(A)
    coeffs, _, _, _ = lstsq(A, F)

    rel_l2, rel_energy = compute_errors(coeffs, nq_val=nq_val)

    elapsed = time.time() - t0

    return {
        "degree": max_degree,
        "n_basis": max_degree + 1,
        "coeffs": coeffs,
        "rel_l2": rel_l2,
        "rel_energy": rel_energy,
        "cond_A": cond_A,
        "time": elapsed,
    }


def plot_solution(result):
    degree = result["degree"]
    coeffs = result["coeffs"]

    x = np.linspace(-1.0, 1.0, 2001)
    u_pred, _, _ = evaluate_solution(coeffs, x)
    u_true = exact(x)

    plt.figure(figsize=(7, 4))
    plt.plot(x, u_true, label="Exact solution")
    plt.plot(x, u_pred, "--", label=f"Traditional Galerkin, degree={degree}")
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_PATH, f"solution_degree_{degree}.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.semilogy(x, np.abs(u_pred - u_true) + 1e-16)
    plt.xlabel("x")
    plt.ylabel("|u_N(x) - u(x)|")
    plt.title(f"Pointwise error, degree={degree}")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_PATH, f"pointwise_error_degree_{degree}.png"), dpi=300)
    plt.close()


def main():
    # You can adjust these degrees.
    # For eps=1e-5, the solution has a sharp internal layer near x=0,
    # so low-degree global polynomials may perform poorly.
    degrees = [2, 4, 6, 8, 10, 12, 16, 20, 24, 32, 40, 50, 64]

    records = []

    for deg in degrees:
        result = solve_for_degree(deg, nq_train=4096, nq_val=4096)
        records.append(result)

        print(
            f"degree={result['degree']:3d}, "
            f"basis={result['n_basis']:3d}, "
            f"rel_L2={result['rel_l2']:.4e}, "
            f"rel_energy={result['rel_energy']:.4e}, "
            f"cond(A)={result['cond_A']:.4e}, "
            f"time={result['time']:.3f}s"
        )

    # Save raw results
    with open(os.path.join(RESULTS_PATH, "traditional_galerkin_results.pkl"), "wb") as f:
        pickle.dump(records, f)

    # Save CSV summary
    csv_path = os.path.join(RESULTS_PATH, "traditional_galerkin_summary.csv")
    with open(csv_path, "w") as f:
        f.write("degree,n_basis,rel_l2,rel_energy,cond_A,time\n")
        for r in records:
            f.write(
                f"{r['degree']},{r['n_basis']},"
                f"{r['rel_l2']},{r['rel_energy']},"
                f"{r['cond_A']},{r['time']}\n"
            )

    # Plot convergence curve
    degrees_arr = np.array([r["degree"] for r in records])
    l2_arr = np.array([r["rel_l2"] for r in records])
    energy_arr = np.array([r["rel_energy"] for r in records])
    cond_arr = np.array([r["cond_A"] for r in records])

    plt.figure(figsize=(7, 4))
    plt.semilogy(degrees_arr, l2_arr, "o-", label="Relative L2 error")
    plt.semilogy(degrees_arr, energy_arr, "s--", label="Relative energy error")
    plt.xlabel("Polynomial degree")
    plt.ylabel("Relative error")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_PATH, "traditional_galerkin_error_vs_degree.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.semilogy(degrees_arr, cond_arr, "o-")
    plt.xlabel("Polynomial degree")
    plt.ylabel("cond(A)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_PATH, "traditional_galerkin_condition_number.png"), dpi=300)
    plt.close()

    # Plot final solution
    plot_solution(records[-1])

    print("\nSaved results to:", RESULTS_PATH)
    print("CSV summary:", csv_path)


if __name__ == "__main__":
    main()