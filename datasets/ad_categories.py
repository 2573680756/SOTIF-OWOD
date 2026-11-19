
SODA_KNOWN_CLASSES = (
    "pedestrian",
    "cyclist",
    "car",
    "truck",
    "bus",
    "tricycle",
)

SODA_CODA_CLASS_NAMES = SODA_KNOWN_CLASSES + ("unknown",)

BDD_KNOWN_CLASSES = (
    "person",
    "rider",
    "car",
    "truck",
    "bus",
)

BDD_CODA_CLASS_NAMES = BDD_KNOWN_CLASSES + ("unknown",)

SODA_CATEGORY_ID_TO_LABEL = {
    1: 0,
    2: 1,
    3: 2,
    4: 3,
    5: 4,
    6: 5,
}

CODA_CATEGORY_ID_TO_SODA_LABEL = {
    1: 0,
    2: 1,
    3: 2,
    4: 3,
    7: 4,
    6: 5,
}

BDD_CATEGORY_NAME_TO_LABEL = {
    "person": 0,
    "rider": 1,
    "car": 2,
    "truck": 3,
    "bus": 4,
}

CODA_CATEGORY_ID_TO_BDD_LABEL = {
    1: 0,
    2: 1,
    3: 2,
    4: 3,
    7: 4,
}
