"""AeroNet backend: airfoil flow solver and HTTP API.

The panel-method solver in this package is a classical stand-in for the NNfoil
physics-informed neural network. The API is written so that a trained PINN can be
registered as a second solver without changing the frontend (see ``aeronet.solvers``).
"""

__version__ = "1.0.0"
