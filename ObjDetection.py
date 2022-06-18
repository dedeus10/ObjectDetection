from doctest import testfile
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import pandas as pd
import os 

class Yolo:
    def __init__(self, path) -> None:
        # Settings
        weights_path = 'yolo-files/yolov3.weights'
        configuration_path = 'yolo-files/yolov3.cfg'
        names_path = 'yolo-files/coco.names'
        self.media_path = path
        show = True

        self.labels = open(names_path).read().strip().split('\n')
        self.probability_minimum = 0.5
        self.threshold = 0.3
        self.font = cv2.FONT_HERSHEY_PLAIN
        self.colours = np.random.uniform(0, 255, size=(len(self.labels), 3))
        self.outDim = (600,800)
       
        
        self.network = cv2.dnn.readNetFromDarknet(configuration_path, weights_path)
        layers_names_all = self.network.getLayerNames()  # list of layers' names
        self.layers_names_output = [layers_names_all[(i - 1)] for i in self.network.getUnconnectedOutLayers()]  # list of layers' names

        #Check media format
        media_type = self.testFormat(self.media_path)

        if(media_type == 'video'):
            #Load video
            self.cap = cv2.VideoCapture(self.media_path)

            if(self.cap.isOpened()):
                print("Video Sucessfully opened!")
                self.mediaSettings = {
                'frames': int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                'fps'   : int(self.cap.get(cv2.CAP_PROP_FPS)),
                'width' : int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                }
            else:
                self.mediaSettings = {
                'frames': None,
                'fps'   : None,
                'width' : None,
                'height': None,
                }
        elif (media_type == 'image'):
            self.cap = cv2.imread(path)
            image_shape = self.cap.shape# Getting image shape
            # height, width, number of channels in image
            self.mediaSettings = {
                'frames': 1,
                'fps'   : None,
                'height' : image_shape[0],
                'width': image_shape[1],
                }
        elif (media_type == 'online'):
            self.cap = cv2.VideoCapture(self.media_path)
            self.mediaSettings = {
                'frames': int(500),
                'fps'   : int(self.cap.get(cv2.CAP_PROP_FPS)),
                'width' : int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                }
        else:
            self.mediaSettings = {
                'frames': None,
                'fps'   : None,
                'width' : None,
                'height': None,
                }

        if(show):
            print("-- MEDIA SPECS --")
            for i in zip(self.mediaSettings.keys(), self.mediaSettings.values()):
                print(i)

    def run_detection(self, show=True):
        pTime = 0   #int to count time to calculate fps
        for nf in range(self.mediaSettings['frames']):
            try:
                # Get current frame
                if(self.mediaSettings['frames'] == 1):
                    self.image = self.cap
                else:
                    _, self.image = self.cap.read()

                self.image = cv2.resize(self.image, self.outDim, interpolation = cv2.INTER_AREA)
                #self.image = cv2.rotate(self.image, cv2.ROTATE_180)
            
                blob = cv2.dnn.blobFromImage(self.image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
                # Slicing blob and transposing to make channels come at the end
                blob_to_show = blob[0, :, :, :].transpose(1, 2, 0)

                # Calculating at the same time, needed time for forward pass
                self.network.setInput(blob)  # setting blob as input to the network
                output_from_network = self.network.forward(self.layers_names_output)

                bounding_boxes = []
                confidences = []
                class_numbers = []
                h = self.outDim[1]
                w = self.outDim[0]
                
                for result in output_from_network:
                    # Going through all detections from current output layer
                    for detection in result:
                        # Getting class for current object
                        scores = detection[5:]
                        class_current = np.argmax(scores)

                        # Getting confidence (probability) for current object
                        confidence_current = scores[class_current]

                        # Eliminating weak predictions by minimum probability
                        if(confidence_current > self.probability_minimum):
                            # Scaling bounding box coordinates to the initial image size
                            # YOLO data format keeps center of detected box and its width and height
                            # That is why we can just elementwise multiply them to the width and height of the image
                            box_current = detection[0:4] * np.array([w, h, w, h])

                            # From current box with YOLO format getting top left corner coordinates
                            # that are x_min and y_min
                            x_center, y_center, box_width, box_height = box_current.astype('int')
                            x_min = int(x_center - (box_width / 2))
                            y_min = int(y_center - (box_height / 2))

                            # Adding results into prepared lists
                            bounding_boxes.append([x_min, y_min, int(box_width), int(box_height)])
                            confidences.append(float(confidence_current))
                            class_numbers.append(class_current)

                results = cv2.dnn.NMSBoxes(bounding_boxes, confidences, self.probability_minimum, self.threshold)

                obj_name = []
                for i in range(len(class_numbers)):
                    obj_name.append(self.labels[int(class_numbers[i])])

                #if(show):
                for i in range(len(bounding_boxes)):
                    if i in results:
                        x, y, w, h = bounding_boxes[i]
                        label = str(self.labels[class_numbers[i]])
                        color = self.colours[i]
                        cv2.rectangle(self.image, (x, y), (x + w, y + h), color, 1)
                        cv2.putText(self.image, label, (x, y + 30), self.font, 2, color, 2)
                #Calculate the FPS
                cTime = time.time()
                fps = 1/(cTime-pTime)
                pTime = cTime

                #Draw FPS 
                cv2.putText(self.image, 'FPS: '+str(int(fps)), (10,70), cv2.FONT_HERSHEY_COMPLEX, 1,
                        (124,0,0), 2 )
                cv2.imshow("Image", self.image)
                k = cv2.waitKey(5) & 0xFF

            except KeyboardInterrupt:
                cv2.destroyAllWindows()
                self.cap.release()
        
        #if (cv2.waitKey(0) & 0xFF == ord('q')):
        #    cv2.destroyAllWindows()
        #    self.cap.release()
        return obj_name, bounding_boxes, confidences, class_numbers, results, self.image

    def testFormat(self, file):
        vid_fm = [".flv", ".avi", ".mp4", ".3gp", ".mov", ".webm", ".ogg", ".qt", ".avchd"]
        img_fm = [".tif", ".tiff", ".jpg", ".jpeg", ".gif", ".png", ".eps", \
          ".raw", ".cr2", ".nef", ".orf", ".sr2", ".bmp", ".ppm", ".heif"]
        filename, file_extension = os.path.splitext(file)
        file_extension = file_extension.lower()

        #print(file_extension)
        if(file_extension in vid_fm):
            media_type = 'video'
        elif(file_extension in img_fm):
            media_type = 'image'
        else:
            media_type = 'NONE'

        if (file.startswith('rtsp')):
            media_type = 'online'

        return media_type


def main():
    test_file = 'test/homeoffice.jpeg'
    test_file = 'test/dogs.MOV'
    test_file = 'rtsp://admin:123456@192.168.1.120:554'
    od = Yolo(path=test_file)
    od.run_detection()



if __name__=='__main__':
    main()