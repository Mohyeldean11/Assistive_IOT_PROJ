import cv2, os,time
# file_path = os.getcwd() + "pic.jpg"
def captureframes()->dict :
    captureobject = cv2.VideoCapture(0)
    if captureobject.isOpened() :
        retval,frame = captureobject.read()
        if retval:
            cv2.imwrite("capture.jpg",frame)
    captureobject.release()

# for i in range(5):
#     captureframes()
#     time.sleep(3)