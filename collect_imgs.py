import os
import cv2

DATA_DIR = './sign lanuage update/sign-language-detector-python/data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 31
dataset_size = 100


start_class = 1


cap = cv2.VideoCapture(0)

for j in range(start_class, number_of_classes):
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))
    
    print('Collecting data for class {}'.format(j))
    
    # Wait for user to press 'q' to start collecting for this class
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break
        
        frame = cv2.flip(frame, 1)  
        cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break
    
   
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break
        
        frame = cv2.flip(frame, 1)  # Apply mirroring when saving as well
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(DATA_DIR, str(j), f'{counter}.jpg'), frame)
        
        counter += 1

cap.release()
cv2.destroyAllWindows()