import cv2
import numpy as np

# Model file paths (ensure these files are in your project folder)
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

# Mean values for model preprocessing
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

# Age and gender categories
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

# Load pre-trained models
print("Loading models...")
try:
    faceNet = cv2.dnn.readNet(faceModel, faceProto)
    ageNet = cv2.dnn.readNet(ageModel, ageProto)
    genderNet = cv2.dnn.readNet(genderModel, genderProto)
    print("Models loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")
    print("Please ensure all model files are in the project folder")
    exit()

def highlightFace(net, frame, conf_threshold=0.7):
    """Detect faces in the frame"""
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    
    # Create blob from frame
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), 
                                  [104, 117, 123], True, False)
    
    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []
    
    # Process detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            faceBoxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), 
                         (0, 255, 0), int(round(frameHeight/150)), 8)
    
    return frameOpencvDnn, faceBoxes

def predictAgeGender(face):
    """Predict age and gender for a face"""
    # Prepare blob for gender prediction
    blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), 
                                  MODEL_MEAN_VALUES, swapRB=False)
    
    # Gender prediction
    genderNet.setInput(blob)
    genderPreds = genderNet.forward()
    gender = genderList[genderPreds[0].argmax()]
    
    # Age prediction
    ageNet.setInput(blob)
    agePreds = ageNet.forward()
    age = ageList[agePreds[0].argmax()]
    
    return gender, age

# Start video capture
print("Starting webcam...")
video = cv2.VideoCapture(0)

if not video.isOpened():
    print("Error: Could not open webcam")
    exit()

padding = 20

print("\nPress 'q' to quit")
print("Real-time detection started...\n")

while True:
    # Read frame from webcam
    hasFrame, frame = video.read()
    if not hasFrame:
        break
    
    # Detect faces
    resultImg, faceBoxes = highlightFace(faceNet, frame)
    
    if not faceBoxes:
        cv2.putText(resultImg, "No face detected", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    # Process each detected face
    for faceBox in faceBoxes:
        # Extract face region with padding
        x1, y1, x2, y2 = faceBox
        face = frame[max(0, y1-padding):min(y2+padding, frame.shape[0]-1),
                    max(0, x1-padding):min(x2+padding, frame.shape[1]-1)]
        
        if face.size == 0:
            continue
        
        # Predict age and gender
        gender, age = predictAgeGender(face)
        
        # Display results on frame
        label = f"{gender}, {age}"
        cv2.putText(resultImg, label, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Print to console
        print(f"Detected: {label}")
    
    # Show the frame
    cv2.imshow("Age and Gender Detection", resultImg)
    
    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
print("\nCleaning up...")
video.release()
cv2.destroyAllWindows()
print("Program ended successfully!")