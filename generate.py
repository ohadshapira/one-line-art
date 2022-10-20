from image import Image
from fourier import Fourier
from plot import Plot
from config import Config

image_1 = Image(Config.INPUT_IMAGE_PATH, shape=Config.OUTPUT_SHAPE)

path_1 = image_1.find_path()

period_1, tup_circle_rads_1, tup_circle_locs_1 = Fourier(n_approx=1000, coord_1=path_1).get_complex_transform()

# plotting
Plot(period_1, tup_circle_rads_1, tup_circle_locs_1, visualize=True).show(
    show_sliders=True)  # fourier_terms=1900,time_term=0.5

# Plot(period_1, tup_circle_rads_1, tup_circle_locs_1, visualize=True).generate_video(type='fourier')

# Plot(period_1, tup_circle_rads_1, tup_circle_locs_1, speed=20, visualize=False).animate(close_after_animation=False)
# Plot(period_1, tup_circle_rads_1, tup_circle_locs_1, speed=20).animate(close_after_animation=False)


pass
