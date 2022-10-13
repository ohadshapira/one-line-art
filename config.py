class Config(dict):
    CANNY_EDGE_DETECTOR = True

    INPUT_IMAGE_PATH = "images/pikachu.png"
    # INPUT_IMAGE_PATH="images/note.png"
    # INPUT_IMAGE_PATH="images/einstein.jpg"
    # INPUT_IMAGE_PATH="images/obama.jpg"

    # OUTPUT_SHAPE=(200, 200)
    OUTPUT_SHAPE = (400, 400)
    DROP_CONTOURS=False
    CONTOURS_THRESHOLD=0.00007