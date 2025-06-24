from signal_processing_utils import (
    impulse_response,
    torch_polyval_flipped,
    torch_polyval,
)
from torch.nn import Module, Parameter
from torch.nn.functional import conv1d
from torchaudio.functional import convolve
import torch


class PhysicalRIRModel(Module):
    """
    This class implements the physical model of the RIR.
    """

    g0: torch.Tensor
    g0_inv: torch.Tensor
    omega: torch.Tensor

    def __init__(
        self,
        Lg: int,
        Lp: int,
        Lh: int,
        force_zeros: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # Length of the RIR
        self.Lh = Lh
        self.Lg = Lg
        self.Lp = Lp

        self.Nfft = min(1 + self.Lp * (self.Lh - 1), 1 << 14)

        # Initialize model parameters
        # g: identity filter
        g = torch.zeros(Lg)
        g[0] = 1
        # a: no decay
        a = torch.tensor(0.0)
        # p: identity filter
        p = torch.zeros(Lp)
        p[0] = 1
        # Energy of the base noise for the rir
        sigma_epsilon_2 = torch.tensor(1.0)
        # Energy of the error
        sigma_w_2 = torch.tensor(1.0)

        if force_zeros:
            # g0: forced zeros of filter g
            self.register_buffer(
                "g0_inv",
                torch.tensor([1, 0, -1], dtype=torch.get_default_dtype()),
            )
            # Ensure g0 is symetric so we save flip in the kernel computation
            Lg0 = Lh if Lh % 2 else Lh - 1
            g0 = (torch.arange(Lg0, dtype=torch.get_default_dtype()) + 1) % 2
            self.register_buffer("g0", g0)
        else:
            self.g0_inv = None
            self.g0 = None

        # Attach parameters to the Module
        # Attached parameters are moved automatically to the device
        # and require gradients
        self.g = Parameter(g)
        self.a = Parameter(a)
        self.p = Parameter(p)
        self.sigma_epsilon_2 = Parameter(sigma_epsilon_2)
        self.sigma_w_2 = Parameter(sigma_w_2)

        self.register_buffer(
            "omega", torch.exp(-2 * 1j * torch.pi * torch.arange(self.Nfft) / self.Nfft)
        )

    @torch.compile()
    def lvecmul_G(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute G @ x, or G @ G0 @ x if zeros are forced.

        Args
        -----
        x: torch.Tensor
            Signal to be filtered, shape (Lh)

        Returns
        -------
        torch.Tensor
            Filtered signal, shape (Lh)
        """
        if self.g0 is not None:
            kernel = conv1d(
                input=self.g0.view(1, 1, -1),
                weight=self.g.view(1, 1, -1),
                padding=self.Lg - 1,
            )
        else:
            kernel = self.g.flip(0).view(1, 1, -1)
        return conv1d(
            input=x.expand(1, 1, -1),
            weight=kernel,
            padding=kernel.shape[-1] - 1,
        ).squeeze()[: x.shape[0]]

    def lvecmul_PE(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute PE @ x

        Args
        -----
        x: torch.Tensor
            Signal to be filtered, shape (Lh)

        Returns
        -------
        torch.Tensor
            Filtered signal, shape (Lh)
        """
        self.P_fourier: torch.Tensor = torch.fft.fft(
            self.p * torch.exp(self.a), self.Nfft
        )
        Y_fourier = torch_polyval_flipped(x, self.P_fourier * self.omega)
        y: torch.Tensor = torch.fft.ifft(Y_fourier, dim=0)[: self.Lh]
        return y.real

    # @torch.compile(mode="reduce-overhead")
    def lvecmul_PEG(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute PEG @ x

        Args
        -----
        x: torch.Tensor
            Signal to be filtered, shape (Lh)
        Returns
        -------
        torch.Tensor
            Filtered signal, shape (Lh)
        """
        return self.lvecmul_PE(self.lvecmul_G(x))

    def trace_PEG_Rh(self, rh: torch.Tensor) -> torch.Tensor:
        """
        Compute the trace tr(PEG Rh PEG.T) needed in the optimization. Rh is diag(rh)

        Args
        -----
        rh: torch.Tensor
            Variance tensor, shape (Lh)

        Returns
        -------
        torch.Tensor
            Trace of the product, scalar
        """
        values = self.P_fourier * self.omega

        # Approximation - Not cutting the sum on G
        # It was tested that the approximation is accurate and much more efficient
        # The approximation becomes exact if rh[-Lg:] = 0
        if self.g0 is not None:
            kernel = conv1d(
                input=self.g0.view(1, 1, -1),
                weight=self.g.view(1, 1, -1),
                padding=self.Lg - 1,
            ).squeeze()
            Gs = torch_polyval(kernel, values)
        else:
            Gs = torch_polyval_flipped(self.g, values)
        Rs = torch_polyval_flipped(rh, values.real**2 + values.imag**2)

        return (Gs.real**2 + Gs.imag**2) @ Rs / self.Nfft

    def lvecmul_G_inv(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute G_inv @ x, or (G @ G0)_inv @ x if zeros are forced.

        Args
        ----
        x: torch.Tensor
            Signal to be filtered, shape (Lh)
        Returns
        -------
        torch.Tensor
            Filtered signal, shape (Lh)
        """
        # 512 should be enough for Lg <= 80
        if self.g0 is not None:
            num = torch.zeros_like(self.g)
            num[: len(self.g0_inv)] = self.g0_inv
            g_inv = impulse_response(self.g, numerator=num, n_fft=512)
        else:
            num = torch.zeros_like(self.g)
            num[0] = 1.0
            g_inv = impulse_response(self.g, numerator=num, n_fft=512)
        kernel = g_inv.flip(0).view(1, 1, -1)

        return conv1d(
            input=x.expand(1, 1, -1),
            weight=kernel,
            padding=kernel.shape[-1] - 1,
        ).squeeze()[: x.shape[0]]

    def lvecmul_PE_inv(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute PE_inv @ x

        Args
        -----
        x: torch.Tensor
            Signal to be filtered, shape (Lh)

        Returns
        -------
        torch.Tensor
            Filtered signal, shape (Lh)
        """
        # We stop the computations of q early
        # if the coefficients are small enough
        # for a certain number of iterations
        threshold = 1e-9
        n_iter = 0
        iter_limit = self.Lp

        q = torch.zeros(self.Lh - 1, device=x.device, dtype=x.dtype)
        q[0] = 1 / self.p[0]
        P = torch.zeros(self.Lh, self.Lh, device=x.device, dtype=x.dtype)
        P[0, 0] = 1.0

        conv = self.p.data.clone()
        for n in range(1, self.Lh - 1):
            conv_len = min(self.Lh - n, len(conv))
            P[n : n + conv_len, n] = conv[:conv_len]
            conv = convolve(conv, self.p[:conv_len])

            s = P[n + 1, 1 : n + 1] @ q[:n]
            q[n] = -s / conv[0]
            if abs(q[n]) < threshold:
                n_iter += 1
                if n_iter > iter_limit:
                    break
            else:
                n_iter = 0

        Q_fourier: torch.Tensor = torch.fft.fft(q * torch.exp(-self.a), self.Nfft)
        Y_fourier = torch_polyval_flipped(x, Q_fourier * self.omega)
        y: torch.Tensor = torch.fft.ifft(Y_fourier, dim=0)[: self.Lh]
        return y.real

    def lvecmul_PEG_inv(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute PEG_inv @ x

        Args
        -----
        x: torch.Tensor
            Signal to be filtered, shape (Lh)

        Returns
        -------
        torch.Tensor
            Filtered signal, shape (Lh)
        """
        return self.lvecmul_G_inv(self.lvecmul_PE_inv(x))
