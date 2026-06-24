"""
my_policy.py
============
Policy module for the Lunar Lander PPO agent (v3).

Loaded by evaluate_agent.py via importlib.
Exposes exactly one function:

    policy_action(params, observation) -> int  (action in {0,1,2,3})

Network architecture (matches train_agent_v3.py Actor exactly)
-----------------------------------------------------------
    Input(8)
      -> Linear(8->256) -> LayerNorm(256) -> GELU
      -> Linear(256->256) -> LayerNorm(256) -> GELU
      -> Linear(256->4)   [argmax for action]

Parameter layout in flat array (total = 70148)
----------------------------------------------
    Linear0.weight   : (256,  8) =  2048   <- row-major, PyTorch (out, in)
    Linear0.bias     : (256,)    =   256
    LayerNorm0.weight: (256,)    =   256   <- scale (gamma)
    LayerNorm0.bias  : (256,)    =   256   <- shift (beta)
    Linear1.weight   : (256,256) = 65536
    Linear1.bias     : (256,)    =   256
    LayerNorm1.weight: (256,)    =   256
    LayerNorm1.bias  : (256,)    =   256
    Linear2.weight   : (  4,256) =  1024
    Linear2.bias     : (  4,)    =     4
    ──────────────────────────────────────
    Total                        = 70148
"""

import numpy as np

HIDDEN_SIZE = 256   # ← updated from 128 to match train_agent_v3.py


# ── Helper functions ──────────────────────────────────────────────────────────

def _gelu(x: np.ndarray) -> np.ndarray:
    """GELU activation — matches PyTorch's default (tanh approximation)."""
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


def _layer_norm(x: np.ndarray, weight: np.ndarray,
                bias: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mean = x.mean()
    var  = x.var()
    return (x - mean) / np.sqrt(var + eps) * weight + bias


# ── Main function ─────────────────────────────────────────────────────────────

def policy_action(params: np.ndarray, observation: np.ndarray) -> int:
    """
    Select a greedy action using the trained actor network.

    Parameters
    ----------
    params      : np.ndarray, shape (70148,)
                  Flat array of actor weights saved by train_agent_v3.py.
    observation : np.ndarray, shape (8,)
                  Current LunarLander-v3 observation.

    Returns
    -------
    int
        Greedy action (0=nothing, 1=left, 2=main engine, 3=right).
    """
    H   = HIDDEN_SIZE
    idx = 0

    W0 = params[idx : idx + H * 8].reshape(H, 8); idx += H * 8
    b0 = params[idx : idx + H];                   idx += H

    ln0_weight = params[idx : idx + H]; idx += H
    ln0_bias   = params[idx : idx + H]; idx += H

    W1 = params[idx : idx + H * H].reshape(H, H); idx += H * H
    b1 = params[idx : idx + H];                   idx += H

    ln1_weight = params[idx : idx + H]; idx += H
    ln1_bias   = params[idx : idx + H]; idx += H

    Wa = params[idx : idx + 4 * H].reshape(4, H); idx += 4 * H
    ba = params[idx : idx + 4];                   idx += 4

    h = observation @ W0.T + b0
    h = _layer_norm(h, ln0_weight, ln0_bias)
    h = _gelu(h)

    h = h @ W1.T + b1
    h = _layer_norm(h, ln1_weight, ln1_bias)
    h = _gelu(h)

    logits = h @ Wa.T + ba
    return int(np.argmax(logits))
