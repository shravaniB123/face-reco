import cv2
import face_recognition
#import face_recognition_models
import numpy as np
import sqlite3
from tkinter import Tk, Label, Button

def get_customer_data(face_encoding):
    conn = sqlite3.connect('customer_database.db')
    c = conn.cursor()
    c.execute("SELECT * FROM customers")
    customers = c.fetchall()
    
    for customer in customers:
        stored_face_encoding = np.frombuffer(customer[2], dtype=np.float64)
        matches = face_recognition.compare_faces([stored_face_encoding], face_encoding)
        if True in matches:
            c.execute("SELECT * FROM transactions WHERE customer_id=?", (customer[0],))
            transactions = c.fetchall()
            conn.close()
            return customer, transactions

    conn.close()
    return None, None

def main():
    video_capture = cv2.VideoCapture(0)
    process_this_frame = True
    #face_recognition_model = face_recognition_models.version.VERSION_1_0

    while True:
        ret, frame = video_capture.read()
        if process_this_frame:
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = small_frame[:, :, ::-1]
            face_locations = face_recognition.face_locations(rgb_small_frame)
            #print("face_locations:", face_locations)
            if face_locations:
                face_encodings = face_recognition.face_encodings(rgb_small_frame, [face_locations[0][0]])
                print("face_encodings:", face_encodings)
            
                for face_encoding in face_encodings:
                    customer, transactions = get_customer_data(face_encoding) 
                    if customer:
                        name = customer[1]
                        transaction_history = "\n".join([t[2] for t in transactions])
                        show_info(name, transaction_history) 
            else:
                print("No faces detected in the frame")
        process_this_frame = not process_this_frame
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

def show_info(name, transaction_history):
    root = Tk()
    root.title("Customer Information")

    name_label = Label(root, text=f"Name: {name}")
    name_label.pack()
    
    history_label = Label(root, text=f"Transaction History:\n{transaction_history}")
    history_label.pack()
    
    survey_label = Label(root, text="Customer Satisfaction Survey")
    survey_label.pack()
    
    def submit_survey():
        print("Survey submitted")
        root.destroy()

    submit_button = Button(root, text="Submit", command=submit_survey)
    submit_button.pack()
    
    root.mainloop()

if __name__ == "__main__":
    main()