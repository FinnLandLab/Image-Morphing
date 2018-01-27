from typing import List
import edge_detect
import numpy as np
from skimage.draw import polygon_perimeter


class Polygon:
    """ A class used to represent a polygon"""

    def __init__(self, pixels: List[edge_detect.Pixel]):
        """

        @param List[edge_detect.Pixel] pixels:
        """
        self.pixels = pixels

    def align_order(self, pixels: List[edge_detect.Pixel]):
        """ Align this polygon's vertex ordering with the given vertex ordering

        @param List[edge_detect.Pixel] pixels:
        """
        # Find the closest pixel to the first one in the given list
        first_px = pixels[0]
        closest_px = min(self.pixels, key=lambda x: first_px.dist_squared(x))
        i = self.pixels.index(closest_px)

        # Come up with two possible orderings, one the reverse of the other
        order = self.pixels[i:] + self.pixels[:i]
        reversed_order = [closest_px] + [x for x in reversed(self.pixels)][:-1]

        # Check which ordering goes about in the same direction as the given list
        amt = min(20, len(order))
        order_sum = sum(order[i].dist_squared(pixels[i]) for i in range(amt))
        reversed_order_sum = sum(reversed_order[i].dist_squared(pixels[i]) for i in range(amt))

        # Set the appropriate one to this polygon's ordering
        if order_sum < reversed_order_sum:
            self.pixels = order
        else:
            self.pixels = reversed_order

    def pad(self, pixels: List[edge_detect.Pixel]):
        """ Pads this polygon to have the same number of pixels as the given list

        @param pixels:
        @return:
        """
        new_pxs = []

        amt = len(pixels) // len(self.pixels)
        extras = len(pixels) - amt * len(self.pixels)

        for px in self.pixels:
            if extras > 0:
                pxs = [px] * (amt + 1)
                extras -= 1
            else:
                pxs = [px] * amt
            new_pxs.extend(pxs)

        self.pixels = new_pxs

    def align(self, other):
        """ Align this Polygon's pixels with the other polygon's pixels

        @param Polygon other:
        """
        if len(self.pixels) < len(other.pixels):
            self.align_order(other.pixels)
            self.pad(other.pixels)
        elif len(self.pixels) > len(other.pixels):
            other.align(self)
        elif len(self.pixels) != 0:
            self.align_order(other.pixels)

    def interpol(self, other, c):
        """ Return a new polygon interpolled between the two polygons

        @param other:
        @param c:
        @return:
        """
        pixels = self.pixels[:]
        for i in range(len(pixels)):
            pixels[i] = pixels[i].interpol(other.pixels[i], c)
        return Polygon(pixels)

    def get_image(self, shape):
        """ Return an image of this polygon

        @return np.ndarray: a dnarray containing the image data
        """
        output = np.zeros(shape).astype(np.uint8)
        r, c = [], []
        for i, px in enumerate(self.pixels):
            r += [px.row]
            c += [px.col]

        rr, cc = polygon_perimeter(r, c, shape, True)
        output[rr, cc] = 255
        return output


def get_edge_polygon(data: np.ndarray) -> Polygon:
    """ Return the pixels in the given image that are in edges, sorted.

    @param numpy.ndarray data: the image data
    @rtype lst of Pixel
    """
    unordered_edges = edge_detect.get_edge_pixels(data)
    edges = [unordered_edges.pop(0)]
    while len(unordered_edges) != 0:
        cur = edges[-1]
        next_ = min(unordered_edges, key=lambda x: cur.dist_squared(x))
        unordered_edges.remove(next_)
        edges.append(next_)

    return Polygon(edges)
