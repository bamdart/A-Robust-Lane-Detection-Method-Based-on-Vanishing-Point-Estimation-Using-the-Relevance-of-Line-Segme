
### HoughLine
DrawHoughLine = True
DEBUG = True

def Hough(src, gray):
    # 二值化抓出一整塊區域
    # Deal = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 159, 15)

    # if(DEBUG): cv2.imshow("Rotate AT", Deal)
    Deal = cv2.blur(gray[top:bottom,left:right], (5,5))

    # 抓邊緣
    Deal = cv2.Canny(Deal, 50, 150)

    if(DEBUG): cv2.imshow("Rotate Canny", Deal)

    # 從邊緣判斷是不是直線
    # rho：距離解析度，越小表示定位要求越準確，但也較易造成應該是同條線的點判為不同線。
    # theta：角度解析度，越小表示角度要求越準確，但也較易造成應該是同條線的點判為不同線。
    # threshold：累積個數閾值，超過此值的線才會存在lines這個容器內。
    # srn：可有可無的距離除數。
    # stn：可有可無的角度除數。
    lines = cv2.HoughLinesP(Deal, rho = 2, theta = np.pi / 180 , threshold = 10 , minLineLength = 20, maxLineGap = 20)
    
    # 沒有直線直接回傳原本的
    if(lines is None):
        return src

    # 直線圖 Debug時顯示
    if(DrawHoughLine):
        lineImg = src.copy()
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(lineImg[top:bottom,left:right],(x1,y1),(x2,y2),(255,0,0),2)
        cv2.imshow("Rotate AT", lineImg)