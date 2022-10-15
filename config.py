class Config(dict):
    CANNY_EDGE_DETECTOR = True

    INPUT_IMAGE_PATH = "images/pikachu.png"
    # INPUT_IMAGE_PATH="images/note.png"
    # INPUT_IMAGE_PATH="images/einstein.jpg"
    # INPUT_IMAGE_PATH="images/obama.jpg"

    OUTPUT_SHAPE=(400, 400)
    # OUTPUT_SHAPE = (400, 400)
    DROP_CONTOURS = True
    CONTOURS_THRESHOLD = 0.00006

    CREATE_VIDEO=True
    VIDEO_INPUT_PATH="video_inputs/"
    VIDEO_OUTPUT_PATH="video_inputs/"
    VIDEO_STEP=0.005

    MAIL_ADDRESS="pyohads@gmail.com"
    MAIL_PASSWORD="Ohad5656105"