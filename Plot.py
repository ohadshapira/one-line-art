import numpy as np

from matplotlib import animation
from matplotlib import pyplot as plt
from matplotlib.patches import ConnectionPatch
from matplotlib.widgets import Slider, Button, RadioButtons

from utils import *


class Plot(object):
    def __init__(self, period, tup_circles_rad, tup_circles_loc, speed=8, visualize=False, background_image=None):
        self.fig = plt.figure(1)

        self.use_background_image = False
        if background_image is not None:
            self.background_image = background_image
            self.use_background_image = True

        # if Config.SHOW_SLIDERS and False:
        #     self.fig.subplots_adjust(bottom=0.25)
        #     # Draw sliders
        #     n_approx_0 = tup_circles_loc[0].shape[0] - 1
        #     n_approx_slider_ax = self.fig.add_axes([0.25, 0.15, 0.65, 0.03], facecolor='lightgoldenrodyellow')
        #     self.n_approx_slider = Slider(n_approx_slider_ax, 'n_approx', 0, n_approx_0, valinit=n_approx_0)
        #
        #     progress_0 = 1
        #     progress_slider_ax = self.fig.add_axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
        #     self.progress_slider = Slider(progress_slider_ax, 'progress', 0, 1, valinit=progress_0)
        #
        #     self.n_approx_slider.on_changed(self.sliders_on_changed)
        #     self.progress_slider.on_changed(self.sliders_on_changed)

        self.period = period
        self.tup_circles_loc = tup_circles_loc
        self.speed = speed
        self.visualize = visualize

        # Two circle lists means we have to draw two images and two sets of circles.
        if len(tup_circles_rad) == 2:
            # 224 is the bottom right subplot
            # 222 is the top right subplot
            # 221 is the top left subplot
            # 223 is the bottom left subplot
            self.axes = [self.fig.add_subplot(int(i)) for i in ("224", "222", "221", "223")]

            # bottom right/top left subplot = circles, bottom left/top right subplot = images
            # axesA=axes[0], axesB=axes[3] connects bottom right subplot to top right subplot
            # axesA=axes[0], axesB=axes[1] connects bottom right subplot to bottom left subplot
            # axesA=axes[2], axesB=axes[3] connects top left subplot to top right subplot
            # axesA=axes[2], axesB=axes[1] connects top left subplot to bottom left subplot
            self.con_patch_tup = tuple(self.get_con_patch((0, 0), (0, 0), axesA, axesB) for (axesA, axesB) in
                                       zip([0] * 2 + [2] * 2, [1, 3] * 2))
            self.add_con_patch(self.con_patch_tup)
            self.axes[1].set_zorder(-1)
            self.axes[3].set_zorder(-1)

            # Points that draws the images
            self.final_points = (self.get_final_point(self.axes[1]), self.get_final_point(self.axes[3]))
            self.x_lim = min(np.amin(tup_circles_loc[0][-1].real), np.amin(tup_circles_loc[1][-1].real)), max(
                np.amax(tup_circles_loc[0][-1].real), np.amax(tup_circles_loc[1][-1].real))
            self.y_lim = max(np.amax(tup_circles_loc[0][-1].imag), np.amax(tup_circles_loc[1][-1].imag)), min(
                np.amin(tup_circles_loc[0][-1].imag), np.amin(tup_circles_loc[1][-1].imag))

        else:
            self.axes = [self.fig.add_subplot(111)]

            # Point that draws the images
            self.final_points = (self.get_final_point(self.axes[0]),)
            self.x_lim = np.floor(np.amin(tup_circles_loc[0][-1].real)), np.ceil(np.amax(tup_circles_loc[0][-1].real))
            self.y_lim = np.floor(np.amax(tup_circles_loc[0][-1].imag)), np.ceil(np.amin(tup_circles_loc[0][-1].imag))

            for ax in self.axes:
                ax.set_xlim(self.x_lim)
                ax.get_xaxis().set_visible(False)
                ax.set_ylim(self.y_lim)
                ax.get_yaxis().set_visible(False)

            if self.use_background_image:
                self.axes[0].imshow(self.background_image, extent=(
                    int(self.x_lim[0]), int(self.x_lim[1]), int(self.y_lim[0]), int(self.y_lim[1])))

        if self.visualize is False:
            circle_lst = list()
            axes = (0, 2)
            for n, circle_rad_lst in enumerate(tup_circles_rad):
                circle_lst.append(list())
                for radius in circle_rad_lst:
                    circle = self.get_circle((0, 0), radius)
                    self.axes[axes[n]].add_patch(circle)
                    circle_lst[n].append(circle)
                # Center circle doesn't move, so remove it!
                circle_lst[n].pop(0)
            self.tup_circles_lst = tuple(circle_lst)

            # Define an action for modifying the line when any slider's value changes

    def sliders_on_changed(self, val):
        self.final_points[0].set_data(self.get_circle_loc_slice(0, 0, int(self.n_approx_slider.val),
                                                                int(self.progress_slider.val * self.period) - 1))
        self.n_text.set_text(
            'Number of Fourier Terms = {fourier_terms}, Progress:{precent}%'.format(
                fourier_terms=int(self.n_approx_slider.val), precent=round(int(self.progress_slider.val * 100), 2)))
        self.fig.canvas.draw_idle()

    @staticmethod
    def get_circle(loc, radius):
        return plt.Circle(loc, np.absolute(radius), alpha=1, fill=False)

    def get_con_patch(self, xyA, xyB, axesA, axesB):
        return ConnectionPatch(xyA=xyA, xyB=xyB,
                               coordsA="data", coordsB="data",
                               axesA=self.axes[axesA], axesB=self.axes[axesB],
                               zorder=25, fc="w", ec="darkblue", lw=2)

    def add_con_patch(self, con_patch_tup):
        self.axes[0].add_artist(con_patch_tup[0])
        self.axes[0].add_artist(con_patch_tup[1])
        self.axes[2].add_artist(con_patch_tup[2])
        self.axes[2].add_artist(con_patch_tup[3])

    def get_final_point(self, axis):
        return axis.plot(0, 0, color='#000000')[0]

    def animate(self, save=False, ani_name=None, ImageMagickLoc=None, close_after_animation=True):
        if self.visualize:
            self.get_visualize()
            update = self.update
            time = self.time
        else:
            update, time = self.get_draw(close_after_animation=close_after_animation, save=save)

        ani = animation.FuncAnimation(self.fig, update, time, interval=1, blit=True, repeat=close_after_animation)
        if save is True and ImageMagickLoc is not None:
            plt.rcParams['animation.convert_path'] = ImageMagickLoc
            writer = animation.ImageMagickFileWriter(fps=100)
            ani.save(ani_name if ani_name else 'gif_1.gif', writer=writer)
        else:
            # TODO(Darius): Figure out a way to get Matplotlib to close the figure nicely after animation is done
            try:
                plt.show()
                plt.axis('off')
            except Exception as e:  # _tkinter.TclError: invalid command name "pyimage10"
                pass

        plt.clf()
        plt.cla()
        plt.close()

    def generate_video(self, type='progress', fourier_terms=None):
        Config.VIDEO_PATH = Config.VIDEO_PATH + return_time_str()
        Config.VIDEO_IMAGES_PATH = Config.VIDEO_PATH + '/images/'
        Config.VIDEO_OUTPUT_PATH = Config.VIDEO_PATH + '/video/'

        os.mkdir(Config.VIDEO_PATH)
        os.mkdir(Config.VIDEO_IMAGES_PATH)
        os.mkdir(Config.VIDEO_OUTPUT_PATH)

        if self.visualize:
            self.get_visualize()
            update = self.update
            time = self.time

        if fourier_terms:
            pass
        else:
            fourier_terms = self.tup_circles_loc[0].shape[0] - 1
        if type == 'progress':
            time_terms = np.arange(0, 1, Config.VIDEO_TIME_STEP)
            for idx, time_term in enumerate(time_terms):
                time_term = max(int(min(self.period * time_term, self.period) - 1), 0)

                self.final_points[0].set_data(self.get_circle_loc_slice(0, 0, fourier_terms, time_term))
                self.n_text.set_text(
                    'Number of Fourier Terms = {fourier_terms}, Progress:{precent}%'.format(fourier_terms=fourier_terms,
                                                                                            precent=round(
                                                                                                time_term / self.period,
                                                                                                2)))
                self.fig.savefig(Config.VIDEO_IMAGES_PATH + str(idx).zfill(4) + '.png')
        else:
            fourier_terms = np.arange(0, self.tup_circles_loc[0].shape[0] - 1, Config.VIDEO_FOURIER_STEP)
            full_time_term = max(int(min(self.period * 1, self.period) - 1), 0)
            for idx, fourier_term in enumerate(fourier_terms):
                self.final_points[0].set_data(self.get_circle_loc_slice(0, 0, fourier_term, full_time_term))
                self.n_text.set_text(
                    'Number of Fourier Terms = {fourier_terms}, Progress:{precent}%'.format(fourier_terms=fourier_term,
                                                                                            precent=100))
                self.fig.savefig(Config.VIDEO_IMAGES_PATH + str(idx).zfill(4) + '.png')

        images_to_video()

        try:
            plt.show()
            plt.axis('off')
        except Exception as e:  # _tkinter.TclError: invalid command name "pyimage10"
            pass

    def show(self, fourier_terms=None, time_term=None):
        if Config.SHOW_SLIDERS:
            self.fig.subplots_adjust(bottom=0.25)
            # Draw sliders
            n_approx_0 = self.tup_circles_loc[0].shape[0] - 1
            n_approx_slider_ax = self.fig.add_axes([0.25, 0.15, 0.65, 0.03], facecolor='lightgoldenrodyellow')
            self.n_approx_slider = Slider(n_approx_slider_ax, 'n_approx', 0, n_approx_0, valinit=n_approx_0)

            progress_0 = 1
            progress_slider_ax = self.fig.add_axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
            self.progress_slider = Slider(progress_slider_ax, 'progress', 0, 1, valinit=progress_0)

            self.n_approx_slider.on_changed(self.sliders_on_changed)
            self.progress_slider.on_changed(self.sliders_on_changed)

        if self.visualize:
            self.get_visualize()
            update = self.update
            time = self.time

        if fourier_terms:
            pass
        else:
            fourier_terms = self.tup_circles_loc[0].shape[0] - 1

        if time_term:
            time_term = int(min(self.period * time_term, self.period) - 1)
        else:
            time_term = -1
            time_term = (self.period - 1)
        self.final_points[0].set_data(self.get_circle_loc_slice(0, 0, fourier_terms, time_term))
        self.n_text.set_text(
            'Number of Fourier Terms = {fourier_terms}, Progress:{precent}%'.format(fourier_terms=fourier_terms,
                                                                                    precent=100 * round(
                                                                                        time_term / self.period,
                                                                                        2)))

        try:
            plt.show()
            plt.axis('off')
        except Exception as e:  # _tkinter.TclError: invalid command name "pyimage10"
            pass

        # plt.clf()
        # plt.cla()
        # plt.close()

    def get_draw(self, close_after_animation, save):
        time = np.arange(0, self.period, self.speed)

        def update(i):
            if close_after_animation and not save and i == time[-1]:
                plt.close()
            else:
                for n_1, circles_tup in enumerate(self.tup_circles_lst):
                    for n_2, circle in enumerate(circles_tup):
                        circle.center = self.get_circle_loc_point(n_1, n_1, circle_idx=n_2, time_idx=i)
                if len(self.tup_circles_lst) == 2:
                    self.final_points[0].set_data(self.get_circle_loc_slice(0, 1, -1, i))
                    self.final_points[1].set_data(self.get_circle_loc_slice(1, 0, -1, i))
                    for con_patch in self.con_patch_tup:
                        con_patch.remove()
                    con_patch_lst = []
                    for ((idx_1, idx_2), (idx_3, idx_4)), (axesA, axesB) in zip(
                            zip([(0, 0)] * 2 + [(1, 1)] * 2, [(0, 1), (1, 0)] * 2), zip([0] * 2 + [2] * 2, [1, 3] * 2)):
                        con_patch_lst.append(self.get_con_patch(self.get_circle_loc_point(idx_1, idx_2, -1, i),
                                                                self.get_circle_loc_point(idx_3, idx_4, -1, i), axesA,
                                                                axesB))
                    self.con_patch_tup = tuple(con_patch_lst)
                    self.add_con_patch(self.con_patch_tup)
                else:
                    self.final_points[0].set_data(self.get_circle_loc_slice(0, 0, -1, i))
            return ([])

        return update, time

    def get_circle_loc_point(self, idx_1, idx_2, circle_idx, time_idx):
        return (
            self.tup_circles_loc[idx_1][circle_idx, time_idx].real,
            self.tup_circles_loc[idx_2][circle_idx, time_idx].imag)

    def get_circle_loc_slice(self, idx_1, idx_2, circle_idx, time_idx):
        return (self.tup_circles_loc[idx_1][circle_idx, :time_idx].real,
                self.tup_circles_loc[idx_2][circle_idx, :time_idx].imag)

    def get_visualize(self):
        self.n_text = self.axes[0].text(0.02, 0.95, 'Number of Points = 0', transform=self.axes[0].transAxes)

        def update(i):
            self.final_points[0].set_data(self.get_circle_loc_slice(0, 0, i, -1))
            self.n_text.set_text('Number of Fourier Terms = %d' % i)
            return ([])

        self.time = np.arange(0, self.tup_circles_loc[0].shape[0], self.speed)
        self.update = update
