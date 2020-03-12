import dlib
import cv2
import math

detectro = dlib.get_frontal_face_detector()
#loding dlib face landmark predictor file
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
font = cv2.FONT_HERSHEY_SIMPLEX
#Function to calculate distance between two point
def cal_distance(x1, y1, x2, y2 ):
    distance = math.sqrt(math.pow(x2 - x1, 2) + math.pow(y2 - y1, 2) * 1.0)
    return distance

cap = cv2.VideoCapture(cv2.CAP_ANY)
if (cap.isOpened() == False):
    print("Error opening video stream or file")

#frame_width = int(cap.get(4))
#frame_height = int(int(cap.get(4)))

while(cap.isOpened):
    ret, frame = cap.read()
    if ret == True:

        #Converting image to gray_scale image
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detectro(gray)
        for face in faces:
            #print(face)
            #x1 = face.left()
            #y1 = face.top()
            #x2 = face.right()
            #y2 = face.bottom()

            # Uncomment out to draw rectangle around the face
            #cv2.rectangle(frame, (x1, y1), (x2, y2), (255,255,0), 4)

            landmarks = predictor(gray, face)
            x_p1 = landmarks.part(0).x
            y_p1 = landmarks.part(0).y
            x_p2 = landmarks.part(27).x
            y_p2 = landmarks.part(27).y
            x_p3 = landmarks.part(16).x
            y_p3 = landmarks.part(16).y

            #Calculating Distance
            #dist_total = cal_distance(x_p1, y_p1, x_p3, y_p3)
            dist_1 = cal_distance(x_p1, y_p1, x_p2, y_p2)
            dist_2 = cal_distance(x_p2, y_p2, x_p3, y_p3)
            #print("[INFO] Total Distance", dist_total)
            #print("[INFO] Distance 1", dist_1)
            #print("[INFO] Distance 2", dist_2)

            #Calculating the ratio between the two distance
            if (dist_1 < dist_2):
                ratio = int((dist_1/dist_2)*100)
                #print("[INFO] I am inside if block", ratio)
            else:
                ratio = int((dist_2/dist_1)*100)
                #print("[INFO] I am inside else block", ratio)

            if ratio <= 70:
                #print("[INFO] Printing...", ratio)
                cv2.putText(frame, 'LOOK INTO THE CAMERA', (125, 255), font, 1, (0,255,255), 5, cv2.LINE_AA)

            cv2.circle(frame, (x_p1, y_p1), 4, (0, 255, 255), -1)
            cv2.circle(frame, (x_p2, y_p2), 4, (0, 255, 255), -1)
            cv2.circle(frame, (x_p3, y_p3), 4, (0, 255, 255), -1)

        cv2.imshow('Display Window', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
