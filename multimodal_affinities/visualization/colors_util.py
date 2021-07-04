import numpy as np
import math
import itertools
import struct


def colors_sampler(colors_count, min_color_val=0.0, max_color_val=1.0):
    """
    Colors generator (RGB tuples)
    :param colors_count: Amount of colors to generate in the palette (colors will be spaced out as much as possible)
    :param min_color_val: Minimum of rgb range to sample from
    :param max_color_val: Maximum of rgb range to sample from
    :return: Generator function for colors. Each next() produces a sample of RGB tuple.
    """
    min_val = min_color_val
    max_val = max_color_val

    # The following calculation tries to create #colors_count combinations of three dimensional discreet vectors.
    # Next we try to figure out how many discreet values we need in each dimension.
    # From combinatorics, we get the equation: y = x^3 - x  where y is color_count
    # We don't repeat values such as (0.25, 0.25, 0.25) more than once, hence the subtraction of x).
    # We solve the polynominal with numpy and round up (as we need "at least" the amount of values).
    roots = np.roots([1.0, 0.0, -1.0, -colors_count])
    vals_per_channel = int(math.ceil(max(roots.real)))
    rgb_vals = np.linspace(min_val, max_val, vals_per_channel)
    palette = list(itertools.product(rgb_vals, repeat=3))
    np.random.shuffle(palette)

    for color in palette:
        yield color


def rgb_tuple_to_hex(color_tuple):
    ''' Converts from (r,g,b) to #RRGGBB in hex '''
    color_str_tokens = ['#']
    for color_dim in color_tuple:
        unnormalized_val = int(round(color_dim * 255))
        hex_str = format(unnormalized_val, '02x')
        color_str_tokens.append(hex_str)
    return ''.join(color_str_tokens)



def rgb_hex_to_tuple(rgbstr_hex):
    ''' Converts from #RRGGBB to (r,g,b) in hex '''
    rgbstr_hex = rgbstr_hex[1:] if rgbstr_hex.startswith('#') else rgbstr_hex
    return struct.unpack('BBB', bytes.fromhex(rgbstr_hex))
