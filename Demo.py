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
top = 280
bottom = -1
left = 1
right = -1

# Filter 扇型角度
startAngle = 10
endAngle = 170

#validation 合格角度
leftMax = 160
leftMin = 125
rightMax = 70
rightMin = 30

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


def LSD(src):
    if(len(src.shape) > 2):
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    # Detect lines in the image
    # return [[x0,y0,x1,y1], [width], []]
    lines = lsd.detect(gray[top:bottom, left:right])

    lines = LSDFilter(lines)  # 濾掉線條和計算strength

    # Draw detected lines in the image
    if(DrawLSDLine):
        for i in range(len(lines)):
            cv2.line(src, (lines[i][0][0], lines[i][0][1]),
                     (lines[i][0][2], lines[i][0][3]), (0, 255, 0), 2)

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
            showImage, (max_strength_point[0], max_strength_point[1]), 8, (204, 204, 51), -1)

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
    circleLine = Line(Point(0, crossPoint.y), Point(cols, crossPoint.y), 1)
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
    # cv2.line(showImage, (0, crossPoint.y),
    #          (639, crossPoint.y), (0, 255, 255), 5)
    for line in lines:
        centerPoint = Point(int((line.p1.x + line.p2.x) / 2),
                            int((line.p1.y + line.p2.y) / 2))
        # cv2.circle(
        #     showImage, (centerPoint.x, centerPoint.y), 10, (255, 255, 0), -1)


    thetas = np.arange(np.pi * startAngle / 180,
                       np.pi * endAngle / 180, 0.05)
    x = crossPoint.x + (cols / 2) * np.cos(thetas)
    y = crossPoint.y + (cols / 2) * np.sin(thetas)
    lines = FilterInArc(startAngle, endAngle, crossPoint, lines)
    testLines = []
    for i in range(len(thetas)):
        testLines.append(Line(crossPoint, Point(x[i], y[i]), 1))
        # cv2.line(showImage, (crossPoint.x, crossPoint.y),
        #          (int(x[i]), int(y[i])), (0, 255, 0), 1)
    lines = Score(lines, testLines)
    ansLine = []
    for line in lines:
        if line.inArc and line.count > 5:
            ansLine.append(line)

    for line in ansLine:
        cv2.line(showImage, (line.p1.x, line.p1.y),
                 (line.p2.x, line.p2.y), (255, 0, 255), 5)

    circleLine = Line(Point(0, crossPoint.y), Point(cols, crossPoint.y), 1)
    thetaRight = -1
    thetaLeft = -1
    if(len(ansLine) >= 1):
        thetaRight = GetAngle(ansLine[0], circleLine)
    if(len(ansLine) >= 2):
        thetaLeft = GetAngle(ansLine[1], circleLine)

    if (thetaLeft < 90 and thetaLeft != -1) or thetaRight > 90:
        temp = thetaLeft
        thetaLeft = thetaRight
        thetaRight = temp

    return thetaLeft, thetaRight


def Distance2(p0, p1):
    return (p0.x - p1.x) * (p0.x - p1.x) + (p0.y - p1.y) * (p0.y - p1.y)


def GetAverageV(q, n=-1):
    if n is - 1 or n > len(q):
        n = len(q)
    x, y = 0, 0
    for i in range(n):
        x += q[len(q) - i - 1].x
        y += q[len(q) - i - 1].y
    x /= n
    y /= n
    return Point(x, y)


def GetVarV(q, n=-1):
    if n is - 1 or n > len(q):
        n = len(q)
    avg = GetAverageV(q, n)
    temp = 0
    for i in range(n):
        temp += Distance2(q[len(q) - i - 1], avg)
    temp /= n
    return math.sqrt(temp)


def Validation(qV, qVTemp, qThetaLeft, qThetaRight, crosspoint, qThetaTemp, kv=5, kn=3):
    if len(qV) is 0:
        qV.append(crosspoint)
        qThetaLeft.append(qThetaTemp.x)
        qThetaRight.append(qThetaTemp.y)
    else:
        if Distance2(crosspoint, GetAverageV(qV)) < kv:
            qV.append(crosspoint)
            qVTemp = []
        else:
            qVTemp.append(crosspoint)
        if qThetaTemp.x != -1:
            if qThetaTemp.x > leftMin and qThetaTemp.x < leftMax:
                qThetaLeft.append(qThetaTemp.x)
        if qThetaTemp.y != -1:
            if qThetaTemp.y > rightMin and qThetaTemp.y < rightMax:
                qThetaRight.append(qThetaTemp.y)

    return qV, qVTemp, qThetaLeft, qThetaRight


def Update(qV, qVTemp, qThetaLeft, qThetaRight, kv=5, kn=3):

    def GetAvg(q, n=-1):
        if n is - 1 or n > len(q):
            n = len(q)
        ans = 0
        for i in range(n):
            ans += q[len(q) - i - 1]
        ans /= n
        return ans

    if len(qVTemp) > kn and GetVarV(qVTemp) < kv:
        qV = qVTemp
        qVTemp = []

    return GetAverageV(qV, kn), Point(GetAvg(qThetaLeft, kn), GetAvg(qThetaRight, kn)), qV, qVTemp


def DrawAns(img, crossPoint, theta):
    def getAnotherPoint(cp, t):
        print("T = ", t)
        y = 360 - cp.y
        if t > 90:
            c = y / math.sin(t * math.pi / 180)
            x = math.sqrt(c * c - y * y)
            return int(cp.x - x)
        else:
            x = y / math.tan(t * math.pi / 180)
            return int(cp.x + x)
    cv2.circle(img, (int(crosspoint.x), int(crosspoint.y)),
               10, (0, 0, 255), -1)
    cv2.line(img, (int(crosspoint.x), int(crosspoint.y)),
             (getAnotherPoint(crossPoint, theta.x), 360), (0, 0, 255), 3)
    cv2.line(img, (int(crosspoint.x), int(crosspoint.y)),
             (getAnotherPoint(crossPoint, theta.y), 360), (0, 0, 255), 3)
    return img


if(__name__ == "__main__"):

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('demonOutput.mp4', fourcc, 20.0, (640, 360))

    cap = cv2.VideoCapture('demoInput.mp4')
    qV, qVTemp, qThetaLeft, qThetaRight = [], [], [], []
    while(1):
    # for i in range(100):
        ret, img = cap.read()
        if(ret == False):
            print('Video Empty')
            break

        img = cv2.resize(img, (cols, rows))

        totalTime = time.time()
        lsdLines = LSD(img)

        crosspoint = CalcLineCrossPoint(img, lsdLines)

        crosspoint = Point(crosspoint[0], crosspoint[1])

        thetaLeft, thetaRight = Filter(img, lsdLines, crosspoint)

        qV, qVTemp, qThetaLeft, qThetaRight = Validation(
            qV, qVTemp, qThetaLeft, qThetaRight, crosspoint, Point(thetaLeft, thetaRight))
        crosspoint, theta, qV, qVTemp = Update(
            qV, qVTemp, qThetaLeft, qThetaRight)

        img = DrawAns(img, crosspoint, theta)
        cv2.imshow('showImage', img)

        # frame = cv2.flip(img, 2)
        # # write the flipped frame
        out.write(img)

        print('fps : ', 1 / (time.time() - totalTime))

        cv2.waitKey(1)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
