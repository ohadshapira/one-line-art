class Config(dict):
    CANNY_EDGE_DETECTOR = True

    INPUT_IMAGE_PATH = "images/pikachu.png"
    # INPUT_IMAGE_PATH="images/note.png"
    # INPUT_IMAGE_PATH = "images/einstein.jpg"
    # INPUT_IMAGE_PATH="images/obama.jpg"

    BACKGROUND_IMAGE = "images/obama.jpg"

    OUTPUT_SHAPE = (400, 400)
    # OUTPUT_SHAPE = (400, 400)

    ERODE = True
    DROP_CONTOURS = True
    CONTOURS_THRESHOLD = 0.00005

    SHOW_SLIDERS = True

    CREATE_VIDEO = True
    VIDEO_FPS = 10
    VIDEO_PATH = "video_inputs/"
    VIDEO_TIME_STEP = 0.005
    VIDEO_FOURIER_STEP = 130

    MAIL_ADDRESS = "pyohads@gmail.com"
    MAIL_PASSWORD = "Ohad5656105"
