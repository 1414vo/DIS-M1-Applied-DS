import colorsys


def hsv_to_rgb(h, s, v):
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"


def color_list(n):
    return [hsv_to_rgb(value / (n + 1), 1.0, 1.0) for value in range(n)]
