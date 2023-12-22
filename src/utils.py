"""!@file utils.py
@brief Miscellaneous utilities.

@details Miscellaneous utilities. Currently includes functions to create
arbitrarily long lists of the different colors.

@author Created by I. Petrov on 15/12/2023
"""
import colorsys


def hsv_to_rgb(h, s, v):
    """! Transforms an HSV color value to an RGB one.
    @param h    The hue of the color.
    @param s    The saturation of the color.
    @param v    The value of the color.

    @return     The corresponding RGB color."""
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"


def color_list(n):
    """! Creates a list of equally spaced colors (in hue terms).
    @param n    The length of the list.

    @return     A list of n colors."""
    return [hsv_to_rgb(value / (n + 1), 1.0, 1.0) for value in range(n)]
