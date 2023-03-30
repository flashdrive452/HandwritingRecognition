import PIL
import TORCHVISION as TORCHVISION
import pandas as pd
import numpy as np
import cv2
import math
import os

from matplotlib import pyplot as plt
from skimage.transform._geometric import TRANSFORMS
from tensorflow.core.framework.dataset_options_pb2 import DATA

# DATA ORGANIZATION
INPUT_FILENAME=[]
OUTPUT_CLASS=[]

for DIRNAME, _, FILENAMES in os.WALK('/KAGGLE/input/nist-CHARACTERS-DATASET/CHARACTERS/TEST_IMAGES'):
    for FILENAME in FILENAMES:
        INPUT_FILENAME.APPEND(FILENAME.split('.')[0])
        OUTPUT_CLASS.APPEND(DIRNAME.split('/')[-1])

TESTDATA = pd.DATAFRAME({'FILENAME':INPUT_FILENAME, 'CLASS':OUTPUT_CLASS})

TESTDATA.to_csv('test.csv')

SAMPLEDATA = pd.DATAFRAME({'FILENAME':INPUT_FILENAME,'CLASS':[0 for _ in range(len(OUTPUT_CLASS))]})
SAMPLEDATA.to_csv('SAMPLE_SUBMISSION.CSV')


# IMAGE SIZING AND SHAPING
def PREPAREIMAGE(IMAGE,req_height):
    if IMAGE.ndim == 3:
        IMAGE=cv2.cvtColor(IMAGE, cv2.COLOR_BGR2GRAY)
    height=IMAGE.SHAPE[0]
    FACTOR=req_height/height
    print("Resized by FACTOR : ",FACTOR)
    return cv2.resize(IMAGE,dsize=None,fx=FACTOR,fy=FACTOR)


# IMAGE BLURRING KERNEL FILTER
def CREATEKERNELFILTER(kernelSize,SIGMA,THETA):
    HALFSIZE=kernelSize//2
    kernel=np.zeros([kernelSize,kernelSize])
    SIGMAX = SIGMA
    SIGMAY = SIGMA * THETA
    for i in range(kernelSize):
        for j in range(kernelSize):
            x = i - HALFSIZE
            y = j - HALFSIZE
            expTerm = np.exp(-((x ** 2) / (2 * (SIGMAX ** 2))) - ((y ** 2) / (2 * (SIGMAY ** 2))))
            kernel[i, j] = (1 / (2 * math.pi * SIGMAX * SIGMAY)) * expTerm
    return kernel


# APPLYING KERNEL FILTERS AND CONTOURS ON IMAGE
def Pre_Processing_Sentence(sentence):
    print("RESIZED SENTENCE: ")
    UPDATED_SENTENCE = PREPAREIMAGE(sentence,50)
    #SHOW_IMAGE(UPDATED_SENTENCE, CMAP='GRAY')
    print("BLURRED SENTENCE: ")
    blurred_sentence=cv2.GAUSSIANBLUR(sentence,(5,5),0)
    #SHOW_IMAGE(blurred_sentence,CMAP='GRAY')

    print("FILTERED SENTENCE: ")
    kernelSize=25
    SIGMA=11
    THETA=7
    MINAREA=150
    kernel=CREATEKERNELFILTER(kernelSize,SIGMA,THETA)
    Filtered_sentence=cv2.filter2D(sentence,-1,kernel,borderType=cv2.BORDER_REPLICATE)
    # SHOW_IMAGE(Filtered_sentence,CMAP='GRAY')
    print("THRES SENTENCE: ")
    THRES_VALUE,Thres_sentence=cv2.threshold(Filtered_sentence,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    Thres_sentence=255-Thres_sentence
    # SHOW_IMAGE(Thres_sentence,CMAP='GRAY')

    cv2.__version__
    components,HIERARCHY=cv2.findContours(Thres_sentence,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    print('NO OF COMPONENTS : ', len(components))
    print("CONTOURED SENTENCE: ")
    # SHOW_IMAGE(cv2.DRAWCONTOURS(sentence, components, -1, (255, 0, 0), 5))

    Words = []
    for contour in components:
        if (cv2.CONTOURAREA(contour) >= MINAREA):
            (x, y, w, h) = cv2.boundingRect(contour)
            Words.APPEND([(x, y, w, h), sentence[y:y + h, x:x + w]])

    print('NO OF COMPONENTS AFTER FILTERING: ', len(Words))
    Words.sort()
    return Words


# Calculating Line Intensity(Horizontal Histogram)
def LINE_SEGMENTATION(IMAGE):
    Sentences=[]
    Line_intensity=[0 for line in range(len(IMAGE))]
    for line in range(len(IMAGE)):
        count=0
        for pixel in range(len(IMAGE[0])):
            if(IMAGE[line][pixel]<128):
                count+=1
        Line_intensity[line]=count
    print("LINE INTENSITY: ")
    print(Line_intensity)
    plt.plot(Line_intensity)
    plt.xticks([])
    plt.show()
    return EVALUATING_THRESHOLD(IMAGE,Line_intensity)


# Evaluating Threshold for Line Segmentation
def EVALUATING_THRESHOLD(IMAGE,Line_intensity):
    Line_Segments=[]
    LINE_SEPERATION=0
    START_FLAG=0
    Zero_count=0
    START_INDEX=0
    End_index=0
    SET_FLAG=0
    for line in range(len(Line_intensity)):
        if (Line_intensity[line] == 0):
            Zero_count += 1
            if (SET_FLAG == 0):
                End_index = line
            SET_FLAG = 1
        else:
            if (SET_FLAG == 1 and START_FLAG == 1):
                SET_FLAG = 0
                Line_Segments.APPEND([START_INDEX, End_index])
                START_INDEX = line
                Line_Segments.APPEND(Zero_count)
                LINE_SEPERATION = LINE_SEPERATION + (Zero_count ** 2)
                Zero_count = 0
            if (START_FLAG == 0):
                START_FLAG = 1
                SET_FLAG = 0
                START_INDEX = line
                Zero_count = 0
        Line_Segments.APPEND([START_INDEX, End_index])
        Line_Threshold = math.sqrt(LINE_SEPERATION) / 6
        print("LINE THRESHOLD : ", Line_Threshold)
        print("LINE SEGMENTS : ", Line_Segments)
        return Segmenting_Lines(IMAGE, Line_Segments, Line_Threshold)


# Segmenting Paragraph into Sentences:
def Segmenting_Lines(IMAGE,Line_Segments,Line_Threshold):
    Sentences=[]
    print(Line_Segments)
    for index in range(1,len(Line_Segments),2):
        if(Line_Segments[index]>Line_Threshold):
            y=Line_Segments[index-1][0]-5
            h=Line_Segments[index-1][1]-Line_Segments[index-1][0]+10
            Sentences.APPEND(IMAGE[y:y+h])
    y=Line_Segments[-1][0]-5
    h=Line_Segments[-1][1]-Line_Segments[-1][0]+10
    Sentences.APPEND(IMAGE[y:y+h])
    return Sentences


# Combining two Mis-segmented Words
def combine_Words(Word1, Word2, sentence):
    WORD = [[]]
    WORD[0].APPEND(Word1[0][0])
    # X-AXIS POSITION
    WORD[0].APPEND(min(Word1[0][1], Word2[0][1]))
    # Y-AXIS POSITION
    WORD[0].APPEND((Word2[0][0] - Word1[0][0]) + Word2[0][2])
    # WIDTH
    WORD[0].APPEND(max(Word1[0][1] + Word1[0][3], Word2[0][1] + Word2[0][3]) - WORD[0][1])
    # HEIGHT
    WORD.APPEND(sentence[WORD[0][1]:WORD[0][1] + WORD[0][3], WORD[0][0]:WORD[0][0] + WORD[0][2]])
    return WORD


# Segmenting sentences into words
def WORD_SEGMENTATION(Words,sentence):
    FINAL_WORDS=[]
    word=[]
    FINAL_FLAG=0
    WORD_SEPERATION_SUM=0
    SEPERATION=[]
    for word_no in range(len(Words)-1):
        DISTANCE = Words[word_no + 1][0][0] - (Words[word_no][0][0] + Words[word_no][0][2])
        SEPERATION.APPEND(DISTANCE)
        WORD_SEPERATION_SUM = WORD_SEPERATION_SUM + DISTANCE
    WORD_AVERAGE_THRESHOLD = math.sqrt(WORD_SEPERATION_SUM / (len(Words) - 1))
    print('WORDS SEPERATION : ', SEPERATION)
    print('AVERAGE THRESHOLD FOR WORD SEPERATION : ',WORD_AVERAGE_THRESHOLD)
    for index in range(len(SEPERATION)):
        if (len(word) == 0):
            word = Words[index]
        if (SEPERATION[index] > WORD_AVERAGE_THRESHOLD):
            FINAL_WORDS.APPEND(word)
            word = []
            FINAL_FLAG = 0
        else:
            word = combine_Words(word, Words[index + 1], sentence)
            FINAL_FLAG = 1
    if (FINAL_FLAG == 0):
        FINAL_WORDS.APPEND(Words[-1])
    else:
        FINAL_WORDS.APPEND(word)
    return FINAL_WORDS


# Evaluating VPP Intensity
def EVALUATING_VPP_INTENSITY(PRE_PROCESSED_BINARY_IMAGE):
    VPP_Intensity=[0 for col in range(len(PRE_PROCESSED_BINARY_IMAGE[0]))]
    for row in range(len(PRE_PROCESSED_BINARY_IMAGE)):
        for col in range(len(PRE_PROCESSED_BINARY_IMAGE[row])):
            if(PRE_PROCESSED_BINARY_IMAGE[row][col]==0):
                VPP_Intensity[col]+=1
    print(VPP_Intensity)
    plt.plot(VPP_Intensity)
    plt.xticks([])
    plt.show()
    return VPP_Intensity


# First Level Character Segmentation Using VPP
def FIRST_LEVEL_CHARACTER_SEGMENTATION_UNDER_VPP(PRE_PROCESSED_BINARY_IMAGE):
    VPP_Intensity=EVALUATING_VPP_INTENSITY(PRE_PROCESSED_BINARY_IMAGE)
    CHARACTER_SEGMENTS=[]
    CHARACTER_SEPERATION= 0
    START_FLAG=0
    Zero_count=0
    s=START_INDEX=0
    End_index=0
    SET_FLAG=0
    for col in range(len(VPP_Intensity)):
        if (VPP_Intensity[col] == 0):
            Zero_count += 1
            if (SET_FLAG == 0):
                End_index = col
            SET_FLAG = 1
        else:
            if (SET_FLAG == 1 and START_FLAG == 1):
                SET_FLAG = 0
                CHARACTER_SEGMENTS.APPEND([START_INDEX, End_index])
                START_INDEX = col
                CHARACTER_SEGMENTS.APPEND(Zero_count)
                CHARACTER_SEPERATION = CHARACTER_SEPERATION +(Zero_count ** 2)
                Zero_count = 0
            if (START_FLAG == 0):
                START_FLAG = 1
                SET_FLAG = 0
                START_INDEX = col
                Zero_count = 0
    CHARACTER_SEGMENTS.APPEND([START_INDEX, End_index])
    CHARACTER_THRESHOLD = math.sqrt(CHARACTER_SEPERATION) / 3
    print("CHARACTER THRESHOLD : ", CHARACTER_THRESHOLD)
    print("CHARACTER SEGMENTS : ", CHARACTER_SEGMENTS)
    return CHARACTER_SEGMENTATION(CHARACTER_SEGMENTS, CHARACTER_THRESHOLD, PRE_PROCESSED_BINARY_IMAGE)


# Segmenting Word into Characters under VPP
def CHARACTER_SEGMENTATION(CHARACTER_SEGMENTS,CHARACTER_THRESHOLD,PRE_PROCESSED_BINARY_IMAGE):
    SEGMENTED_CHARACTERS=[]
    for index in range(1,len(CHARACTER_SEGMENTS),2):
        if (CHARACTER_SEGMENTS[index] > CHARACTER_THRESHOLD):
            x = CHARACTER_SEGMENTS[index - 1][0]
            y = 0
            Touching = 0
            w = CHARACTER_SEGMENTS[index - 1][1] - CHARACTER_SEGMENTS[index - 1][0]
            h = len(PRE_PROCESSED_BINARY_IMAGE)
            if (w > 0.675 * h):
                Touching = 1
            SEGMENTED_CHARACTERS.APPEND([[x, y, w, h], PRE_PROCESSED_BINARY_IMAGE[y:y + h, x:x + w], Touching])
        x = CHARACTER_SEGMENTS[-1][0]
        y = 0
        Touching = 0
        w = CHARACTER_SEGMENTS[-1][1] - CHARACTER_SEGMENTS[-1][0]
        h = len(PRE_PROCESSED_BINARY_IMAGE)
        if (w > 0.675 * h):
            Touching = 1
        SEGMENTED_CHARACTERS.APPEND([[x, y, w, h], PRE_PROCESSED_BINARY_IMAGE[y:y + h, x:x + w], Touching])
        return SEGMENTED_CHARACTERS


# Evaluating VPP and TDP Average Intensity
def GENERATE_VPP_AND_TDP_AVERAGE(SEGMENT):
    IMAGE=SEGMENT[1]
    #SHOW_IMAGE(IMAGE,CMAP='GRAY')
    VPP_Intensity=[0 for col in range(len(IMAGE[0]))]
    for row in range(len(IMAGE)):
        for col in range(len(IMAGE[row])):
            if(IMAGE[row][col]==0):
                VPP_Intensity[col]+=1
    print(VPP_Intensity)
    plt.plot(VPP_Intensity)
    plt.xticks([])
    plt.show()

    TDP_Intensity = [0 for col in range(len(IMAGE[0]))]
    for col in range(len(IMAGE[0])):
        for row in range(len(IMAGE)):
            if (IMAGE[row][col] == 0):
                TDP_Intensity[col] = len(IMAGE) - row
                break
    print(TDP_Intensity)
    plt.plot(TDP_Intensity)
    plt.xticks([])
    plt.show()
    AVERAGE_INTENSITY = np.ADD(TDP_Intensity, VPP_Intensity)
    print(AVERAGE_INTENSITY)
    plt.plot(AVERAGE_INTENSITY)
    plt.xticks([])
    plt.show()
    return AVERAGE_INTENSITY


# Connected Components Segmentation
def TOUCHING_CHARACTER_SEGMENTATION(SEGMENT, AVERAGE_INTENSITY,FINAL_SEGMENTED_CHARACTERS):
    AVERAGE_THRESHOLD=20
    TOUCHING_CHARACTERS_BREAKPOINTS=[]
    col=0
    while(col<len(AVERAGE_INTENSITY)):
        if(col==0):
            while(col<len(AVERAGE_INTENSITY) and AVERAGE_INTENSITY[col]<AVERAGE_THRESHOLD):
                col+=1
        if(col<len(AVERAGE_INTENSITY) and AVERAGE_INTENSITY[col]<AVERAGE_THRESHOLD):
            MIN_VALUE=AVERAGE_INTENSITY[col]
            min_point=col
            while(col<len(AVERAGE_INTENSITY) and AVERAGE_INTENSITY[col]<AVERAGE_THRESHOLD):
                if (AVERAGE_INTENSITY[col] < MIN_VALUE):
                    MIN_VALUE = AVERAGE_INTENSITY[col]
                min_point = col
                col += 1
                if (col < len(AVERAGE_INTENSITY)):
                    TOUCHING_CHARACTERS_BREAKPOINTS.APPEND(min_point)
            col += 1
        print("CHARACTERS BREAK POINTS",TOUCHING_CHARACTERS_BREAKPOINTS)
        if (len(TOUCHING_CHARACTERS_BREAKPOINTS) == 0):
            REQUIRED_FURTHER_SEGMENTATION(SEGMENT,AVERAGE_INTENSITY, FINAL_SEGMENTED_CHARACTERS)
        else:
            x_point = 0
            y_point = SEGMENT[0][1]
            height = SEGMENT[0][3]
            for BREAK_POINT in TOUCHING_CHARACTERS_BREAKPOINTS:
                width = BREAK_POINT - x_point
                if (width > 0.8 * height):
                    REQUIRED_FURTHER_SEGMENTATION([[x_point,y_point, width, height], SEGMENT[1][y_point:
                                                    y_point + height,x_point:x_point + width], 1],
                                                  AVERAGE_INTENSITY[x_point:x_point + width ], FINAL_SEGMENTED_CHARACTERS)
                else:
                    FINAL_SEGMENTED_CHARACTERS.APPEND([[x_point, y_point, width, height], SEGMENT[1][y_point:
                                                     y_point + height, x_point:x_point + width], 0])
                x_point = BREAK_POINT
            width = SEGMENT[0][2] - x_point
            if (width > 0.8 * height):
                REQUIRED_FURTHER_SEGMENTATION([[x_point, y_point, width, height], SEGMENT[1][y_point:
                                                y_point + height,x_point:x_point + width], 1],
                                                  AVERAGE_INTENSITY[x_point:x_point + width], FINAL_SEGMENTED_CHARACTERS)
            else:
                FINAL_SEGMENTED_CHARACTERS.APPEND([[x_point, y_point, width, height], SEGMENT[1][y_point:
                                                    y_point + height, x_point:x_point + width], 0])


# Further Required Segmentation on Connected Components
def REQUIRED_FURTHER_SEGMENTATION(IMAGE_SEGMENT,AVERAGE_INTENSITY,FINAL_SEGMENTED_CHARACTERS):
    Index_Limit=int(0.4*IMAGE_SEGMENT[0][3])
    MIN_VALUE=AVERAGE_INTENSITY[Index_Limit]
    BREAK_POINT=Index_Limit
    for col in range(Index_Limit,len(AVERAGE_INTENSITY)- Index_Limit):
        if (AVERAGE_INTENSITY[col] < MIN_VALUE):
            MIN_VALUE = AVERAGE_INTENSITY[col]
            BREAK_POINT = col

    x_point = 0
    y_point = IMAGE_SEGMENT[0][1]
    height = IMAGE_SEGMENT[0][3]
    width = BREAK_POINT - x_point
    x_point = BREAK_POINT
    width = IMAGE_SEGMENT[0][2] - x_point


# Data Loader
class DATASET(DATA.DATASET):
    def __init__(self,CSV_PATH,IMAGES_PATH,TRANSFORM=None):
        self.TRAIN_SET=pd.READ_CSV(CSV_PATH)
        self.TRAIN_PATH=IMAGES_PATH
        self.TRANSFORM=TRANSFORM

    def __len__(self):
        return len(self.TRAIN_SET)

    def __getitem__(self,idx):
        FILE_NAME=self.TRAIN_SET.iloc[idx][1]+'.png'
        LABEL=self.TRAIN_SET.iloc[idx][2]
        img=PIL.image.open(os.PATH.join(self.TRAIN_PATH,FILE_NAME))
        if self.TRANSFORM is not None:
            img=self.TRANSFORM(img)
        return img,LABEL


# Defining Transforms and Parameters:
PARAMS = {'BATCH_SIZE': 16, 'shuffle': True}
epochs = 6
LEARNING_RATE=1e-3
TRANSFORM_TRAIN = TRANSFORMS.Compose([TRANSFORMS.Resize((224,224)),TRANSFORMS.RANDOMAPPLY(
    [TORCHVISION.TRANSFORMS.RANDOMROTATION(10),TRANSFORMS.RANDOMHORIZONTALFLIP()],0.7),
    TRANSFORMS.ToTensor()])


