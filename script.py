from PIL import Image
import numpy as np
import Polygon

data1 = np.array(Image.open('test1.png'))
data2 = np.array(Image.open('test2.png'))

edge1_pxs = Polygon.get_edge_polygon(data1)
edge2_pxs = Polygon.get_edge_polygon(data2)

edge1_pxs.align(edge2_pxs)

for i, c in enumerate(np.linspace(0, 1, 20)):
    interpol = edge1_pxs.interpol(edge2_pxs, c)
    Image.fromarray(interpol.get_image(data1.shape)).save('out/{i:03d}_interpol_{c:}.png'.format(i=i, c=c))
