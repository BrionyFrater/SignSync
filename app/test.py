import cv2


#make camera
camera = cv2.VideoCapture(1)

# def generate_frames():

#     while True:
#         #get frame
#         success, frame = camera.read()

#         if not success:
#             break
#         else:
#             #might need to change to png
#             result, encoded_image = cv2.imencode('.jpg', frame)
#             final = encoded_image.tobytes()

#         yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# generate_frames()

currentframe = 0

while True:
    success, frame = camera.read()

    cv2.imshow("Output", frame)

    cv2.imwrite('frame' + str(currentframe) + '.jpg', frame)

    def save_my_img(frame_count, img):
        path = 'C:/OpenCV/Scripts/Images'
        name = f'/img_title_{currentframe}.jpg'
        cv2.imwrite(os.path.join(path, name), img)

    currentframe += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
#         

camera.release()
cv2.destroyAllWindows()
