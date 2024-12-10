def calculate_indices(red, green, blue):
    """Calculate vegetation indices."""
    rcc = red / (red + green + blue)
    gcc = green / (red + green + blue)
    bcc = blue / (red + green + blue)
    return {"rcc": rcc, "gcc": gcc, "bcc": bcc}
