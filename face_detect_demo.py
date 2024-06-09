import cv2
import face_recognition

# load known face encodings and names
known_face_encodings = []
known_face_names = []

#Load known faces and their names here
image_file_paths = [
    "C:/Users/spbha/OneDrive/Desktop/final_year/face-reco/face-reco/diff image/regular_customers/images (2).jpg",
    "C:/Users/spbha/OneDrive/Desktop/final_year/face-reco/face-reco/diff image/regular_customers/n.jpg",
    "C:/Users/spbha/OneDrive/Desktop/final_year/face-reco/face-reco/diff image/non-regular_customers/s.jpg",
    "C:/Users/spbha/OneDrive/Desktop/final_year/face-reco/face-reco/diff image/regular_customers/images (51).jpg",
    "C:/Users/spbha/OneDrive/Desktop/final_year/face-reco/face-reco/diff image/non-regular_customers/v1.jpg",
    "C:/Users/spbha/OneDrive/Desktop/final_year/face-reco/face-reco/diff image/non-regular_customers/Venkatrao kadam.jpg",
    "C:/Users/spbha/OneDrive/Desktop/final_year/face-reco/face-reco/diff image/non-regular_customers/v7.jpg"
    # Add more image file paths here
]

# Extract face encodings for each known person
for image_file_path in image_file_paths:
    image = face_recognition.load_image_file(image_file_path)
    known_face_encodings.append(face_recognition.face_encodings(image)[0])
    known_face_names.append("Regular customer"  if "regular" in image_file_path else "Non-regular customer")

# Initialize webcam
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()

    # Find all face locations in the current frame
    if frame is None:
        print("Error: frame is None")
        continue
    face_locations = face_recognition.face_locations(frame) # returns a list of tuples (top, right, bottom, left)
    face_encodings = face_recognition.face_encodings(frame, face_locations) # returns a list of face encodings (one for each face in the image)

    # Loop through each face found in the frame
    for(top, right, bottom, left), face_encodings in zip(face_locations, face_encodings):
        # See if the face is a match for the known face(s) in the database
        matches = face_recognition.compare_faces(known_face_encodings, face_encodings)
        name = "New Customer"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        # Draw a box around the face and label with the name
        cv2.rectangle(frame, (left,top), (right, bottom), (0,0,255), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

    # Display the resulting frame
    cv2.imshow("Video", frame)

    # Break the loop when the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close OpenCV windows
video_capture.release()
cv2.destroyAllWindows()




