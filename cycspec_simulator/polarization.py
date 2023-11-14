def validate_stokes(I, Q=None, U=None, V=None):
    """
    Check that arrays representing Stokes parameters have compatible shapes,
    and that either all Stokes parameters or only I are present.
    Return a boolean `full_stokes` indicating which of these is true,
    and the common shape of the arrays.
    """
    full_stokes = False
    if Q is not None or U is not None or V is not None:
        if Q is None or U is None or V is None:
            raise ValueError(
                "Supply either all Stokes parameters (I, Q, U, V) or I only."
            )
        if Q.shape != I.shape or U.shape != I.shape or V.shape != I.shape:
            raise ValueError(
                "Shapes of arrays do not match: "
                f"I.shape = {I.shape}, Q.shape = {Q.shape}, "
                f"U.shape = {U.shape}, V.shape = {V.shape}"
            )
        full_stokes = True
    return full_stokes, I.shape

def coherence_to_stokes(AA, BB, CR, CI, feed_poln):
    """
    Convert coherence data to Stokes parameters, using the provided feed polarization
    (either "LIN" or "CIRC").
    """
    if feed_poln == "LIN":
        # Linearly polarized feed
        I = AA + BB
        Q = AA - BB
        U = 2*CR
        V = 2*CI
    elif feed_poln == "CIRC":
        # Circularly polarized feed
        I = AA + BB
        Q = 2*CR
        U = 2*CI
        V = AA - BB
    else:
        raise ValueError(f"Unrecognized feed polarization '{feed_poln}'.")

    return I, Q, U, V
