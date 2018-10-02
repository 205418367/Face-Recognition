

class Face:
    # face bounding boxes
    def __init__(self, x1, y1, x2, y2, confidence, img):
        self.img = img
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.confidence = confidence
        self.embedding = None

        # self.predict_proba = None
        # self.predict_candidates = None
        # self.predict_candidates_proba = None

    def data(self):
        return { k:v for k, v in self.__dict__.items() if k != 'img'}

