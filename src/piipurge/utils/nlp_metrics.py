
from torch import Tensor
from typing import Tuple, List
from sentence_transformers import SentenceTransformer


ENCODER_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2") 


def get_text_similarity(text1: str | List[str], text2: str | List[str]) -> Tensor:
    """
    Calculates cosine similarity between two strings.

    Args:
        text1: first text value
        text2: second text value

    Returns:
        a Tensor value measuring cosine similarity between the two single 
        string values or lists of strings.
    """
    
    text1_embed = ENCODER_MODEL.encode(text1)
    text2_embed = ENCODER_MODEL.encode(text2)
    similarity = ENCODER_MODEL.similarity(text1_embed, text2_embed)

    return similarity
