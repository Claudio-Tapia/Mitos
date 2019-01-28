import cv2
from sys import platform

class Params:
    _instance = None
    # Singleton class
    def __new__(cls, *args, **kwargs):
        if not isinstance(cls._instance, cls):
            cls._instance = object.__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        self.blobDetectorParams = cv2.SimpleBlobDetector_Params()

        self.blobDetectorParams.filterByArea = True
        self.blobDetectorParams.minArea = 120
        self.blobDetectorParams.maxArea = 2000

        self.blobDetectorParams.filterByColor = False

        self.blobDetectorParams.minConvexity = 0.3
        self.blobDetectorParams.minInertiaRatio = 0.03
        self.blobDetectorParams.minDistBetweenBlobs = 20

        self.blobDetectorParams.minThreshold = 30
        self.blobDetectorParams.maxThreshold = 130

        self.blobDetectorParams.thresholdStep = 25

        self.candidates_size = 80
        self.model_input_size = 80
        self.model_epoch = 40

        if platform == 'win32':
            self.basedir = "C:/Users/PelaoT/Desktop/Practica/dataset/"
        elif platform == 'linux':
            self.basedir = "/home/facosta/dataset/"

        self.normHeStainDir = self.basedir + "normalizado/heStain/"
        self.saveCandidatesWholeImgDir = self.basedir + "train/print/"
        self.saveCutCandidatesDir = self.basedir + "train/candidates/"
        self.saveMitosisPreProcessed = self.basedir + "train/mitosis/preProcessed/"
        self.saveCutMitosDir = self.basedir + "train/mitosis/"
        self.saveTestCandidates = self.basedir + "eval/no-mitosis/"
        self.saveTestMitos = self.basedir + "eval/mitosis/"
        self.candidatesTrainingJsonPath = self.basedir + "anotations/trainCandidates.json"
        self.candidatesTestJsonPath = self.basedir + "anotations/test_cand.json"
        self.mitosAnotationJsonPath = self.basedir + "anotations/MitosAnotations.json"
