import cv2
import numpy as np
import time
import math
import numba as nb
rows, cols = 360, 640

DrawLSDLine = True
LSD_strength_threshold = 6

DrawCrossPoint = True

# 只需要偵測此區域內的車道線
top = 270
bottom = -10
left = 120
right = -120

# Create default parametrization LSD
lsd = cv2.createLineSegmentDetector(0)

# 回傳(x0, y0, x1, y1, strength)


# Filter parameter
dt = 30
Phit = 30


class Point(object):
    x = 0
    y = 0

    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y


class Line(object):
    def __init__(self, p1, p2, width):
        self.p1 = p1
        self.p2 = p2
        self.width = width
        self.count = 0
        self.inArc = False


def LSDFilter(inputlines):
    lines = []
    for i in range(len(inputlines[0]) - 1, -1, -1):
        strength = LSDLineLength(inputlines[0][i]) / inputlines[1][i]
        if(strength > LSD_strength_threshold):
            lines.append([[inputlines[0][i][0][0] + left, inputlines[0][i][0][1] + top,
                           inputlines[0][i][0][2] + left, inputlines[0][i][0][3] + top, strength]])
    return np.array(lines, dtype=np.float32)

# 計算line strength, 直接取兩個abs，不然pow和sqrt太久了


@nb.jit(nopython=True)
def LSDLineLength(lines):
    # x0 = int(round(lines[0][0]))
    # y0 = int(round(lines[0][1]))
    # x1 = int(round(lines[0][2]))
    # y1 = int(round(lines[0][3]))
    # return math.sqrt(math.pow(x1-x0,2) + math.pow(y1-y0,2))
    return abs(lines[0][2] - lines[0][0])+abs(lines[0][3]-lines[0][1])


def LSD(showImage, src):
    if(len(src.shape) > 2):
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    # Detect lines in the image
    # return [[x0,y0,x1,y1], [width], []]
    lines = lsd.detect(gray[top:bottom, left:right])

    lines = LSDFilter(lines)  # 濾掉線條和計算strength

    # Draw detected lines in the image
    # if(DrawLSDLine):
    #     for i in range(len(lines)):
    #         cv2.line(showImage, (lines[i][0][0], lines[i][0][1]),
    #                  (lines[i][0][2], lines[i][0][3]), (0, 0, 255), 5)

    return lines

# 計算兩條線的交點


def GetCrossPoint(LineA, LineB):
    ka = (LineA[3] - LineA[1]) / (LineA[2] - LineA[0])  # 求出LineA斜率
    kb = (LineB[3] - LineB[1]) / (LineB[2] - LineB[0])  # 求出LineB斜率

    x = (ka*LineA[0] - LineA[1] - kb*LineB[0] + LineB[1]) / (ka - kb)
    y = (ka*kb*(LineA[0] - LineB[0]) + ka*LineB[1] - kb*LineA[1]) / (ka - kb)
    return [x, y]


def CalcLineCrossPoint(showImage, lines):
    strength_img = np.zeros((rows, cols))

    max_strength_point = [0, 0, 0]

    for i in range(len(lines)):
        for j in range(len(lines)):
            if(i != j):
                crossPoint = GetCrossPoint(lines[i][0], lines[j][0])
                if(crossPoint[0] > 0 and crossPoint[0] < cols and crossPoint[1] > 0 and crossPoint[1] < rows):
                    crossPoint[0] = int(crossPoint[0])
                    crossPoint[1] = int(crossPoint[1])
                    # + strngth 在該point
                    strength_img[crossPoint[1], crossPoint[0]
                                 ] += lines[i][0][4] + lines[j][0][4]

                    # 存 strength value 最大的 point
                    if(strength_img[crossPoint[1], crossPoint[0]] > max_strength_point[2]):
                        max_strength_point[0] = crossPoint[0]
                        max_strength_point[1] = crossPoint[1]
                        max_strength_point[2] = strength_img[crossPoint[1],
                                                             crossPoint[0]]

    cv2.imshow('crosspoint_img', strength_img)

    if(DrawCrossPoint):
        cv2.circle(
            showImage, (max_strength_point[0], max_strength_point[1]), 10, (255, 0, 0), -1)

    return max_strength_point


def Trans(lines):
    ret = []
    for line in lines:
        ret.append(Line(Point(line[0][0], line[0][1]),
                        Point(line[0][2], line[0][3]), line[0][4]))
    return ret


def GetLinePara(line):
    line.a = line.p1.y - line.p2.y
    line.b = line.p2.x - line.p1.x
    line.c = line.p1.x * line.p2.y - line.p2.x * line.p1.y
    return line


def GetAngle(lineA, lineB):
    GetLinePara(lineA)
    GetLinePara(lineB)
    angle = ((lineA.a * lineB.a) + (lineA.b * lineB.b)) / (math.sqrt(pow(lineA.a,
                                                                         2) + pow(lineA.b, 2)) * math.sqrt(pow(lineB.a, 2) + pow(lineB.b, 2)))
    return (math.acos(angle) * 180 / np.pi)


def FilterInArc(startAngle, endAngle, crossPoint, lines):
    circleLine = Line(Point(0, crossPoint.y), Point(640, crossPoint.y), 1)
    for line in lines:
        tmpAngle = GetAngle(line, circleLine)
        if tmpAngle > startAngle and tmpAngle < endAngle:
            line.inArc = True
    return lines


def GetDistance(line, testLine):
    cenPt = Point((line.p1.x + line.p2.x) / 2,
                  (line.p1.y + line.p2.y) / 2)
    distance = (abs(testLine.a * cenPt.x + testLine.b * cenPt.y + testLine.c) /
                math.sqrt(pow(testLine.a, 2) + pow(testLine.b, 2)))
    return distance


def Score(lines, testLines):
    score = []
    for testLine in testLines:
        for line in lines:
            angle = GetAngle(line, testLine)
            distance = GetDistance(line, testLine)
            if angle < Phit and distance < dt:
                line.count += 1
    return lines


def Filter(showImage, lines, crossPoint):
    lines = Trans(lines)
    cv2.line(showImage, (0, crossPoint.y),
             (639, crossPoint.y), (0, 255, 255), 5)
    for line in lines:
        centerPoint = Point(int((line.p1.x + line.p2.x) / 2),
                            int((line.p1.y + line.p2.y) / 2))
        # cv2.circle(
        #     showImage, (centerPoint.x, centerPoint.y), 10, (255, 255, 0), -1)

    startAngle = 33
    endAngle = 160
    thetas = np.arange(np.pi * startAngle / 180,
                       np.pi * endAngle / 180, 0.05)
    x = crossPoint.x + 320 * np.cos(thetas)
    y = crossPoint.y + 320 * np.sin(thetas)
    lines = FilterInArc(startAngle, endAngle, crossPoint, lines)
    testLines = []
    for i in range(len(thetas)):
        testLines.append(Line(crossPoint, Point(x[i], y[i]), 1))
        cv2.line(showImage, (crossPoint.x, crossPoint.y),
                 (int(x[i]), int(y[i])), (0, 255, 0), 1)
    lines = Score(lines, testLines)
    ansLine = []
    for line in lines:
        if line.inArc and line.count > 5:
            ansLine.append(line)
    for line in ansLine:
        cv2.line(showImage, (line.p1.x, line.p1.y),
                 (line.p2.x, line.p2.y), (255, 0, 255), 7)
    circleLine = Line(Point(0, crossPoint.y), Point(640, crossPoint.y), 1)
    thetaRight = -1
    thetaLeft = -1
    if(len(ansLine) >= 1):
        thetaRight = GetAngle(ansLine[0], circleLine)
    if(len(ansLine) >= 2):
        thetaLeft = GetAngle(ansLine[1], circleLine)
    return thetaLeft, thetaRight


if(__name__ == "__main__"):
    cap = cv2.VideoCapture('test.mp4')
    while(1):
        print('=======================================================')
        ret, img = cap.read()
        if(ret == False):
            print('Video Empty')
            break

        img = cv2.resize(img, (cols, rows))
        showImage = img.copy()

        totalTime = time.time()
        cv2.imshow('origin', img)

        s = time.time()
        lsdLines = LSD(showImage, img)
        print('LSDLines spend time', time.time() -
              s, " ,line amount :", len(lsdLines))

        s = time.time()
        crosspoint = CalcLineCrossPoint(showImage, lsdLines)
        print('CalcLineCrossPoint spend time', time.time() -
              s, " ,location(x,y,strength) :", crosspoint)
        crosspoint = Point(crosspoint[0], crosspoint[1])
        print('Detect spend time', time.time() - totalTime)
        thetaLeft, thetaRight = Filter(showImage, lsdLines, crosspoint)
        cv2.imshow('showImage', showImage)
        cv2.waitKey(1)
