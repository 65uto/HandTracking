import cv2
import mediapipe as mp
import time
import matplotlib.pyplot as plt
cap = cv2.VideoCapture(0)
#image = cv2.imread('E:\Anapatch_folder\Coding\Python\VisuslMouse\img\hand.jpeg')
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mpDraw = mp.solutions.drawing_utils
pointX4 = 0
pointX8 = 0
pointY4 = 0
pointY8 = 0

while True:
    success, image = cap.read()
    image_rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    plt.imshow(image_rgb)
    result = hands.process(image_rgb)
    result_hands = result.multi_hand_landmarks
    
    if result_hands:
        lmlist = []
        for landmarks in result_hands:
            for id ,lm in enumerate(landmarks.landmark):
                h, w, c = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.rectangle(image,(cx,cy), (cx+10,cy+10), (255,0,0), 1)
                lmlist.append([id,cx,cy])
                
                if id == 4:
                    cv2.circle(image,(cx,cy), 25, (20,0,25), cv2.FILLED)
                    pointX4 = lmlist[id][1]
                    pointY4 = lmlist[id][2]
                elif id == 8:
                    cv2.circle(image,(cx,cy), 25, (255,255,255), cv2.FILLED)
                    pointX8 = lmlist[id][1]
                    pointY8 = lmlist[id][2]

                """ Access Line Point"""
                if (int(pointX4-pointX8) < int(pointX8-pointX4)) and int(pointX8-pointX4) >= 0:
                    if (pointX8-pointX4) < 20 and (pointX8-pointX4)>=0:
                        #print(pointX8-pointX4)
                        cv2.line(image,(pointX8,pointY8),(pointX4,pointY4),(255,0,5),4)
                        show = ((pointX8-pointX4)/20) * 100
                        print(show)
                        cv2.putText(image,str(int(show)), (pointX4+10,pointY4+10) ,0, 2.5,(255,100,0), 4)

                                    
                mpDraw.draw_landmarks(image,landmarks, mp_hands.HAND_CONNECTIONS)
               
                
    
    if cv2.waitKey(1) & 0xFF == ord('e'):
        break

    cv2.imshow('Display', image)

cap.release()
cv2.destroyAllWindows() 
#plt.show()
    
