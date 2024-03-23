import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.PlotModule import LivePlot

# Video yakalama
cap = cv2.VideoCapture("video1.mp4")

# Yüz ağı algılayıcı oluşturma
detector = FaceMeshDetector()

# Grafik çizme
plotY = LivePlot(540, 360, [10, 60])

# İlgili nokta listesi
idList = [22, 23, 24, 26, 110, 157, 158, 159, 160, 161, 130, 243]

# Oran listesi ve sayaçlar
ratioList = []
blick_counter = 0
counter = 0
color = (0, 0, 255)

while True:
    # Frame okuma
    success, img = cap.read()
    if not success:
        break

    # Yüz ağı tespiti
    img, faces = detector.findFaceMesh(img, draw=False)

    if faces:
        face = faces[0]

        # İlgili noktaları çizme
        for id in idList:
            cv2.circle(img, face[id], 5, (255, 0, 0), cv2.FILLED)

        # Gözler arası mesafe hesaplama
        lengthVer = detector.findDistance(face[159], face[23])[0]
        lengthHor = detector.findDistance(face[130], face[243])[0]

        # Oran hesaplama
        ratio = int((lengthVer / lengthHor) * 100)
        ratioList.append(ratio)

        # Son 3 oranın ortalaması
        if len(ratioList) > 3:
            ratioList.pop(0)
        ratioAVG = sum(ratioList) / len(ratioList)

        # Göz kırpma sayısı hesaplama
        if ratioAVG < 35 and counter == 0:
            blick_counter += 1
            color = (0, 255, 0)
            counter = 1
        if counter != 0:
            counter += 1
            if counter > 10:
                counter = 0
                color = (0, 0, 255)

        # Ekran üzerinde bilgileri çizme
        cvzone.putTextRect(img, f'Blink Count: {blick_counter}', (50, 100), colorR=color)
        imgPlot = plotY.update(ratioAVG, color)
        img = cv2.resize(img, (540, 360))
        imgStack = cvzone.stackImages([img, imgPlot], 2, 1)

    # Görüntüyü gösterme
    cv2.imshow("img", imgStack)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Video yakalama serbest bırakma ve pencereleri kapatma
cap.release()
cv2.destroyAllWindows()
