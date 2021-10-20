from .ctcdecoder import beam_search as beam_search_native
import numpy as np

__all__ = ["beam_search"]

def beam_search(probs: np.ndarray, alphabet: str, beam_size: int = 100, lm_model = None, lm_alpha = 0.9, lm_beta = 0.0001):
    return beam_search_native(probs, alphabet, beam_size, lm_model, lm_alpha, lm_beta)
