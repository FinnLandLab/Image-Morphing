from typing import List

import numpy as np
import skimage.feature


class Pixel:
    """ A class used to represent a pixel in an image"""

    def __init__(self, row: int, col: int):
        """ Initialize a pixel obj at the given coordinates

        @param int row: row coordinate
        @param int col: column coordinate
        """
        self.row = row
        self.col = col

    def dist_squared(self, other) -> float:
        """ Return the distance squared between this pixel and another pixel

        @param Pixel other: another pixel
        @return float: the distance squared between the pixels
        """
        return (self.row - other.row)**2 + (self.col - other.col) ** 2

    def interpol(self, other, c: float):
        """ Return a pixel in the interpolated between this pixel and other

        @param Pixel other:
        @param float c: between 0 and 1 inclusive
        @rtype: Pixel
        """
        c_prime = 1 - c
        row = round(c * self.row + c_prime * other.row)
        col = round(c * self.col + c_prime * other.col)
        return Pixel(row, col)


def get_edge_matrix(data: np.ndarray) -> np.ndarray:
    """ Return an numpy ndarray with the edges in the image labelled with
    1, and 0 otherwise
    """
    return skimage.feature.canny(data)


def get_edge_pixels(data: np.ndarray) -> List[Pixel]:
    """ Return the pixels in the given image that are in edges

    @param numpy.ndarray data: the image data
    @rtype lst of Pixel
    """
    edge_matrix = get_edge_matrix(data)
    it = np.nditer(edge_matrix, flags=['multi_index'])
    edges = []
    while not it.finished:
        index = it.multi_index
        if edge_matrix[index] == 1:
            row = index[0]
            col = index[1]
            px = Pixel(row, col)
            edges += [px]
        it.iternext()
    return edges


