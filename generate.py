from Image import Image
from Fourier import Fourier
from Plot import Plot
from Config import Config

image_1 = Image(Config.INPUT_IMAGE_PATH, shape=Config.OUTPUT_SHAPE)


path_1 = image_1.sort()
# path_2 = im_2.sort()
# path_3 = im_3.sort()
# path_4 = im_4.sort()

period_1, tup_circle_rads_1, tup_circle_locs_1 = Fourier(n_approx=1000, coord_1=path_1).get_circles()
# period_2, tup_circle_rads_2, tup_circle_locs_2 = Fourier(n_approx = 1000, coord_1 = path_2).get_circles(mode=2)
# period_3, tup_circle_rads_3, tup_circle_locs_3 = Fourier(n_approx = 1000, coord_1 = path_3, coord_2 = path_4).get_circles()
# period_4, tup_circle_rads_4, tup_circle_locs_4 = Fourier(coord_1 = path_5).get_circles()

##Plot(period_1, tup_circle_rads_1, tup_circle_locs_1, speed = 200).plot(save = True, ani_name = 'image_1.gif', ImageMagickLoc = 'C:\Program Files\ImageMagick-7.0.8-Q16/magick.exe')
Plot(period_1, tup_circle_rads_1, tup_circle_locs_1, speed=20, visualize=True).plot(close_after_animation=False)
# Plot(period_2, tup_circle_rads_2, tup_circle_locs_2, speed = 800).plot(close_after_animation = False)
# Plot(period_3, tup_circle_rads_3, tup_circle_locs_3, speed = 8).plot(close_after_animation = False)
# Plot(period_4, tup_circle_rads_4, tup_circle_locs_4, visualize = True).plot(close_after_animation = False)

pass
