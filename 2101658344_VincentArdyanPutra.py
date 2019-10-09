import cv2
import os
import numpy as np

def get_path_list(root_path):
    return os.listdir(root_path)
    '''
        To get a list of path directories from root path

        Parameters
        ----------
        root_path : str
            Location of root directory
        
        Returns
        -------
        list
            List containing the names of the sub-directories in the
            root directory
    '''

def get_class_names(root_path, train_names):
    face_list = [] 
    class_list = [] 

    for idx, train_name in enumerate(train_names):
        full_name_path = root_path + '/' + train_name
        
        for image_path in os.listdir(full_name_path):
            
            full_image_path = full_name_path + '/' + image_path 
           
            face_list.append(full_image_path)
            class_list.append(idx)

    return face_list, class_list
    '''
        To get a list of train images path and a list of image classes id

        Parameters
        ----------
        root_path : str
            Location of images root directory
        train_names : list
            List containing the names of the train sub-directories
        
        Returns
        -------
        list
            List containing all image paths in the train directories
        list
            List containing all image classes id
    '''

def get_train_images_data(image_path_list):
    train_img = []
    for img_list in image_path_list:
        img = cv2.imread(img_list)

        train_img.append(img)

    return train_img
    '''
        To load a list of train images from given path list

        Parameters
        ----------
        image_path_list : list
            List containing all image paths in the train directories
        
        Returns
        -------
        list
            List containing all loaded train images
    '''

def detect_faces_and_filter(image_list, image_classes_list=None):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    
    filtered_face_list = [] 
    filtered_class_list = [] 
    filtered_face_list_location = []
    
    for index, img in enumerate(image_list):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        detected_faces = face_cascade.detectMultiScale(img, 1.2, 5)

        if(len(detected_faces) < 1 or len(detected_faces) > 1):
            continue

        for x, y, w, h in detected_faces:
            img_face = img[y : y + h, x : x + w]

            filtered_face_list.append(img_face)
            if(image_classes_list != None):
                filtered_class_list.append(image_classes_list[index])
   
            filtered_face_list_location.append([x,y,w,h])
        

    return filtered_face_list, filtered_face_list_location, filtered_class_list

        
    '''
        To detect a face from given image list and filter it if the face on
        the given image is more or less than one

        Parameters
        ----------
        image_list : list
            List containing all loaded images
        image_classes_list : list, optional
            List containing all image classes id
        
        Returns
        -------
        list
            List containing all filtered and cropped face images in grayscale
        list
            List containing all filtered faces location saved in rectangle
        list
            List containing all filtered image classes id
    '''

def train(train_face_grays, image_classes_list):
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(train_face_grays, np.array(image_classes_list))

    return face_recognizer
    '''
        To create and train classifier object

        Parameters
        ----------
        train_face_grays : list
            List containing all filtered and cropped face images in grayscale
        image_classes_list : list
            List containing all filtered image classes id
        
        Returns
        -------
        object
            Classifier object after being trained with cropped face images
    '''

def get_test_images_data(test_root_path, image_path_list):
    test_img = []
    for img_test_list in os.listdir(test_root_path):
        full_image_path = test_root_path + '/' + img_test_list
        
        img = cv2.imread(full_image_path)

        test_img.append(img)

    return test_img
    '''
        To load a list of test images from given path list

        Parameters
        ----------
        test_root_path : str
            Location of images root directory
        image_path_list : list
            List containing all image paths in the test directories
        
        Returns
        -------
        list
            List containing all loaded test images
    '''

def predict(classifier, test_faces_gray):
    prediction_list = []

    for img in test_faces_gray:
       img_class_predict, _ = classifier.predict(img)

       prediction_list.append(img_class_predict)

    return prediction_list 
    '''
        To predict the test image with classifier

        Parameters
        ----------
        classifier : object
            Classifier object after being trained with cropped face images
        train_face_grays : list
            List containing all filtered and cropped face images in grayscale

        Returns
        -------
        list
            List containing all prediction results from given test faces
    '''

def draw_prediction_results(predict_results, test_image_list, test_faces_rects, train_names):
    drawn_result = []

    for index, result in enumerate(test_image_list):
        x, y, w, h = test_faces_rects[index]

        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 1)
       
        text = train_names[predict_results[index]]
       
        cv2.putText(result, text, (x, y - 2), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1) 
 
        drawn_result.append(result)    

    return drawn_result

    '''
        To draw prediction results on the given test images

        Parameters
        ----------
        predict_results : list
            List containing all prediction results from given test faces
        test_image_list : list
            List containing all loaded test images
        test_faces_rects : list
            List containing all filtered faces location saved in rectangle
        train_names : list
            List containing the names of the train sub-directories

        Returns
        -------
        list
            List containing all test images after being drawn with
            prediction result
    '''

def combine_results(predicted_test_image_list):
    
    return np.hstack(predicted_test_image_list)

    '''
        To combine all predicted test image result into one image

        Parameters
        ----------
        predicted_test_image_list : list
            List containing all test images after being drawn with
            prediction result

        Returns
        -------
        ndarray
            Array containing image data after being combined
    '''

def show_result(image):
    cv2.imshow('Result', image)
    cv2.waitKey(0)
    '''
        To show the given image

        Parameters
        ----------
        image : ndarray
            Array containing image data
    '''

'''
You may modify the code below if it's marked between

-------------------
Modifiable
-------------------

and

-------------------
End of modifiable
-------------------
'''
if __name__ == "__main__":
    '''
        Please modify train_root_path value according to the location of
        your data train root directory

        -------------------
        Modifiable
        -------------------
    '''
    train_root_path = "dataset/train"
    '''
        -------------------
        End of modifiable
        -------------------
    '''
    
    train_names = get_path_list(train_root_path)
    image_path_list, image_classes_list = get_class_names(train_root_path, train_names)
    train_image_list = get_train_images_data(image_path_list)
    train_face_grays, _, filtered_classes_list = detect_faces_and_filter(train_image_list, image_classes_list)
    classifier = train(train_face_grays, filtered_classes_list)

    '''
        Please modify test_image_path value according to the location of
        your data test root directory

        -------------------
        Modifiable
        -------------------
    '''
    test_root_path = "dataset/test"
    '''
        -------------------
        End of modifiable
        -------------------
    '''

    test_names = get_path_list(test_root_path)
    test_image_list = get_test_images_data(test_root_path, test_names)
    test_faces_gray, test_faces_rects, _ = detect_faces_and_filter(test_image_list)
    predict_results = predict(classifier, test_faces_gray)
    predicted_test_image_list = draw_prediction_results(predict_results, test_image_list, test_faces_rects, train_names)
    final_image_result = combine_results(predicted_test_image_list)
    show_result(final_image_result)