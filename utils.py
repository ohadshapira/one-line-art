from moviepy.editor import *
import os
from config import Config


def images_to_video():
    images_path = Config.VIDEO_INPUT_PATH

    fps = 10
    image_files = [os.path.join(images_path, img)
                   for img in sorted(os.listdir(images_path))]

    clip = ImageSequenceClip(image_files, fps=fps)

    video_name = '{video_path}{video_name}.mp4'.format(video_path=Config.VIDEO_OUTPUT_PATH,
                                                        video_name=Config.INPUT_IMAGE_PATH.split("/")[-1][:-4])
    clip.write_videofile(video_name, verbose=False, logger=None)
