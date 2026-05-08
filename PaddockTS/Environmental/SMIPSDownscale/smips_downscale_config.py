"""Configuration for SMIPS soil moisture downscaling."""

from attrs import frozen


@frozen
class SMIPSDownscaleConfig:
    """Configuration for SMIPS downscaling via convex optimization.

    Attributes
    ----------
    lambda_smoothness : float
        Regularization weight for spatial smoothness (Laplacian term).
        Higher values produce smoother results. Default 0.1.
    solver : str
        CVXPY solver to use. Options: 'SCS', 'OSQP', 'ECOS'. Default 'SCS'.
    max_iters : int
        Maximum solver iterations. Default 5000.
    max_gap_days : int
        Maximum temporal gap (days) when matching SMIPS to S2 observations.
        If no SMIPS within this range, timestep is skipped. Default 1.
    use_terrain : bool
        Include TWI and HLI terrain features in the prior model.
        Requires terrain data to be downloaded. Default True.
    verbose : bool
        Print solver progress. Default False.
    """

    lambda_smoothness: float = 0.5
    solver: str = 'SCS'
    max_iters: int = 5000
    max_gap_days: int = 1
    use_terrain: bool = True
    verbose: bool = True
