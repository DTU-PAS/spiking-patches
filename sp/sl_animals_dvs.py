DATASET_NAME = "sl-animals-dvs"

HEIGHT = 128
WIDTH = 128

ID_TO_CLASS = {
    1: "cat",
    2: "dog",
    3: "camel",
    4: "cow",
    5: "sheep",
    6: "goat",
    7: "wolf",
    8: "squirrel",
    9: "mouse",
    10: "dolphin",
    11: "shark",
    12: "lion",
    13: "monkey",
    14: "snake",
    15: "spider",
    16: "butterfly",
    17: "bird",
    18: "duck",
    19: "zebra",
}

NUM_CLASSES = len(ID_TO_CLASS)

VAL_USERS = {
    "user12_indoor",
    "user13_indoor",
    "user22_sunlight",
    "user31_imse",
    "user32_imse",
    "user50_dc",
    "user51_dc",
    "user52_dc",
    "user53_dc",
}

# This is the test set used by TORE:
# https://github.com/bald6354/tore_volumes/blob/main/code/processAnimalsdataset.m
TEST_USERS = {
    "user14_indoor",
    "user15_indoor",
    "user16_indoor",
    "user17_indoor",
    "user23_sunlight",
    "user24_sunlight",
    "user33_imse",
    "user34_imse",
    "user54_dc",
    "user55_dc",
    "user56_dc",
    "user57_dc",
    "user58_dc",
    "user59_dc",
}
