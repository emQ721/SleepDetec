import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.PlotModule import LivePlot

cap=cv2.VideoCapture("video1.mp4")

detector = FaceMeshDetector()
plotY=LivePlot(540,360,[10,60])

idList =[22,23,24,26,110,157,159,160,161,130,243]
ratioList=[]
blick_counter=0
counter=0
color=(0,0,255)


while True:
    success ,img=cap.read()
    if not success:
        break

    img , faces=detector.findFaceMesh(img,draw=False)

    if faces:
        face=faces[0]

        for id in idList:
            cv2.circle(img,face[id],5,(255,0,0),cv2.FILLED)

        lengthVer=detector.findDistance(face[159],face[23])[0]
        lengthHor=detector.findDistance(face[130],face[243])[0]

        ratio = int((lengthVer/lengthHor)*100)
        ratioList.append(ratio)

        if len(ratioList) > 3:
            ratioList.pop(0)
        ratioAVG=sum(ratioList)/len(ratioList)

        if ratioAVG <35 and counter ==0:
            blick_counter+=1
            color = (0,255,0)
            counter =1
        if counter !=0:
            counter+=1
            if counter >10:
                counter =0
                color = (0,0,255)

        cvzone.putTextRect(img,f"Blink Count:{blick_counter}",(50,100),colorR=color)

        imgPlot = plotY.update(ratioAVG,color)
        img = cv2.resize(img,(540,360))
        imgStack=cvzone.stackImages([img,imgPlot],2,1)

    cv2.imshow("Sleep Detector",imgStack)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
