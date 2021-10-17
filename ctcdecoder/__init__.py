from .ctcdecoder import beam_search as beam_search_native
import numpy as np

__all__ = ["beam_search"]

def beam_search(probs: np.ndarray, alphabet: str, beam_size: int = 100) -> tuple[list[str], list[list[int]]]:
    return beam_search_native(probs, alphabet, beam_size)
