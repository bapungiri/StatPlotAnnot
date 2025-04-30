def adjust_lightness(color, amount=0.5):
    import colorsys

    import matplotlib.colors as mc

    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    c = colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])
    return mc.to_hex(c)


def colors_sd(amount=1):
    return [
        adjust_lightness("#424242", amount=amount),
        adjust_lightness("#eb4034", amount=amount),
    ]


def colors_sd_light(amount=1):
    return [
        adjust_lightness("#707070", amount=amount),
        adjust_lightness("#f18179", amount=amount),
    ]


def colors_mab(amount=1):
    return [
        adjust_lightness("#424242", amount=amount),  # Unstruc
        adjust_lightness("#eb4034", amount=amount),  # Struc
    ]
