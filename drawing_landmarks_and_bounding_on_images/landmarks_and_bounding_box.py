#importing the required libraries
from mtcnn import MTCNN
import cv2
import glob
import os

#definning a function to draw circle
def cv2_circle(image_rgb,landmark_coordinates,thickness=1,radius=1,color = (0, 0, 255)):
    image = cv2.circle(image_rgb, landmark_coordinates, radius, color, thickness)
    return image

#definning a function to draw rectangle for a bounding box
def cv2_rect(image_rgb,start_point,end_point,thickness = 2,color = (0, 255, 0)):
    image_rgb = cv2.rectangle(image_rgb, start_point, end_point, color, thickness)
    return image_rgb

#definning a function to read the image, perform face detection on it using mtcnn, draw bounding box and landmarks of the faces detected
def image_read_save(image_path,input_folder_path,output_folder_path):
    #reading the image
    image_array=cv2.imread(image_path,cv2.IMREAD_UNCHANGED)
    image_rgb=image_array.copy()
    #uncomment this if you want to convert image to BGR format
    # image_rgb=cv2.cvtColor(image_array,cv2.COLOR_RGB2BGR)
    #creating object for MTCNN class
    face_detector=MTCNN()
    #detecting the faces using detect_faces method
    detected_faces=face_detector.detect_faces(image_rgb)
    print(len(detected_faces))
    #iterating through each face mtcnn detected
    for index in range(len(detected_faces)):
        bounding_box=detected_faces[index]['box']
        #drawing circle for each landmark of the face
        for landmark_coordinates in detected_faces[index]['keypoints'].values():
            image_rgb=cv2_circle(image_rgb,landmark_coordinates)
        start_point=(bounding_box[0],bounding_box[1])
        end_point=(bounding_box[0]+bounding_box[2],bounding_box[1]+bounding_box[3])
        #drawing bounding box for face
        cv2_rect(image_rgb,start_point,end_point)
    #saving the image
    cv2.imwrite(image_path.replace(input_folder_path,output_folder_path),image_rgb)
    #displaying the image
    cv2.imshow("IMAGE",image_rgb)
    cv2.waitKey(0)
    #releasing all the resources allocated
    cv2.destroyAllWindows()


#input images folder path
input_folder_path="input_images"
#output images folder path
output_folder_path='output_images'
os.makedirs(output_folder_path,exist_ok=True)

#getting all the jpeg images 
images_paths=glob.glob(input_folder_path+"/*.jpeg")
for image_path in images_paths:
    #calling image_read_save function
    image_read_save(image_path,input_folder_path,output_folder_path)
    
