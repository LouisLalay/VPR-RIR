from torch import Tensor
import numpy as np
import torch


def stabilize_filter(denominator: Tensor) -> Tensor:
    """Ensure all the poles of the filter are inside the unit circle"""
    poles = np.roots(denominator.cpu().detach().numpy())
    changed = False
    for i, val in enumerate(poles):
        if np.abs(val) >= 1:
            changed = True
            poles[i] = 1 / val
    if changed:
        return Tensor(np.poly(poles)).to(denominator)
    return denominator


def torch_polyval(p: Tensor, x: Tensor) -> Tensor:
    """
    Compute the polynomial evaluation of p at x using Horner's method.

    Args
    -----
    p: Tensor
        The polynomial coefficients, shape (order)
    x: Tensor
        The input signal, shape (any)
    Returns
    -------
    Tensor
        The polynomial evaluation, shape x_like

    """
    y = torch.zeros_like(x)
    for coeff in p:
        y = y * x + coeff
    return y


def torch_polyval_flipped(p: Tensor, x: Tensor) -> Tensor:
    """
    Compute the polynomial evaluation of p at x using Horner's method.
    Compared to torch_polyval, p is flipped.

    Args
    -----
    p: Tensor
        The polynomial coefficients, shape (order)
    x: Tensor
        The input signal, shape (any)
    Returns
    -------
    Tensor
        The polynomial evaluation, shape x_like

    """
    order = p.shape[0]
    y = torch.zeros_like(x)
    for k in range(order):
        y = y * x + p[order - k - 1]
    return y


def inverse_g(g: Tensor, n_fft: int = 1024) -> Tensor:
    """
    Compute the inverse IR of g if g is stable.

    Args
    -----
    g: Tensor
        Impulse response g, shape (Lg)

    Returns
    -------
    Tensor
        Inverse of g, shape (n_fft)
    """
    return torch.fft.irfft(1 / torch.fft.rfft(g, n_fft)).real


def linear_regression(x: Tensor, y: Tensor) -> tuple[float, float]:
    """
    Returns the coefficients of the linear regression line y_hat = ax + b
    that best fits the data points (x, y) in the least squares sense.

    Args
    -----
    x: Tensor
        The abscissa values, shape (N)
    y: Tensor
        The ordinate values, shape (N)
    Returns
    -------
    tuple[Tensor, Tensor]
        The coefficients of the linear regression line (a, b)
    """
    N = x.shape[0]
    s_x = x.sum()
    s_y = y.sum()
    x = x.to(y)
    a = (N * x @ y - s_x * s_y).item() / (N * x @ x - s_x**2).item()
    b = (s_y - a * s_x).item() / N

    return a, b
