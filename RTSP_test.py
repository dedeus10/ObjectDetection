import cv2
import sys
'''
@brief: Test RTSO communication with camera
'''
def main():
    cap = cv2.VideoCapture('rtsp://admin:123456@192.168.1.120:554') #IP Camera
    
    while(True):
        ret, frame = cap.read()
        frame=cv2.resize(frame, (960, 540))
        cv2.imshow('Capturing',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): #click q to stop capturing
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    sys.exit(main())