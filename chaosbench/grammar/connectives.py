"""
Grammar connectives — composing atoms into deeper systems.

Post-MVP: AffineConjugacy will allow depth-1 compositions.
"""


class AffineConjugacy:
    """Affine conjugacy connective (post-MVP).

    Transforms an atom f into h⁻¹ ∘ f ∘ h where h is affine.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "AffineConjugacy is a post-MVP feature. "
            "See grammar/connectives.py for planned API."
        )
