from PIL import Image
import numpy as np
import Polygon
import os

image1_path = 'shape1.png'
image2_path = 'shape2.png'

morph_quality = 20

directory = "out/{}-{}/".format(image1_path[:image1_path.index('.')], image2_path[:image2_path.index('.')])
if not os.path.exists(directory):
    os.makedirs(directory)

print("Loading images...")
data1 = np.array(Image.open(image1_path))
print('Loaded {}'.format(image1_path))
data2 = np.array(Image.open(image2_path))
print('Loaded {}'.format(image2_path))

print("Converting images to polygons...")
polygon1 = Polygon.get_edge_polygon(data1)
print("Converted {} to a polygon with {} vertexes".format(image1_path, len(polygon1.pixels)))
polygon2 = Polygon.get_edge_polygon(data2)
print("Converted {} to a polygon with {} vertexes".format(image2_path, len(polygon2.pixels)))

print("Aligning the polygons...")
polygon1.align(polygon2)

print("Saving the morphed images...")
for i, c in enumerate(np.linspace(0, 1, morph_quality)):
    name = '{i:03d}_interpol.png'.format(i=i)
    interpol = polygon1.interpol(polygon2, c)
    Image.fromarray(interpol.get_image(data1.shape)).save(directory + name)
