import matplotlib.pyplot as plt
import numpy as np
points = np.random.rand(100,2) #random
from scipy.spatial import Voronoi, voronoi_plot_2d
vor = Voronoi(points)
fig = voronoi_plot_2d(vor)
fig = voronoi_plot_2d(vor, show_vertices=False, line_colors='orange',
                line_width=3, line_alpha=0.6, point_size=2)
plt.show()