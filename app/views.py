import os
from app import app
from flask import render_template, request, flash, redirect, url_for, jsonify, send_from_directory, Response
from werkzeug.utils import secure_filename
from app.forms import SearchForm, UploadForm
import cv2
import numpy as np 
import mediapipe as mp
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K


import json

model=  load_model('app/1501.h5')
actions= np.array(['Are you hungry','Do you have any pets','How are you',
                   'Where is the library','My favorite color is green',
                   'What is your favorite color',"No, I don't have any pets",
                   'Yes I could go for a snack'])
mp_holistic = mp.solutions.holistic 
mp_drawing = mp.solutions.drawing_utils 

def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS)
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) 
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) 
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
# Function to draw styled landmarks on the image
def draw_styled_landmarks(image, results):
    # Draw face landmarks with specific styling
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                               mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                               mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1))

    # Draw pose landmarks with specific styling
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                               mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                               mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2))

    # Draw left hand landmarks with specific styling
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                               mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                               mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2))

    # Draw right hand landmarks with specific styling
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                               mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                               mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))
    
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert image to RGB
    image.flags.writeable = False  # Make image read-only to improve performance
    results = model.process(image)  # Perform detection
    image.flags.writeable = True  # Make image writable again
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert image back to BGR
    return image, results
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])
def predict(vid):
    # 1. New detection variables
    sequence = []
    sentence = []
    predictions = []
    threshold = 0.7
    frame_count = 0
    frame_interval = 4  # Adjust this value based on your preference
    cap = cv2.VideoCapture(vid)
    # Set mediapipe model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():

            # Read feed
            ret, frame = cap.read()

            # Make detections
            image, results = mediapipe_detection(frame, holistic)
            #print(results)

            # Draw landmarks
            draw_styled_landmarks(image, results)

            # Increment frame count
            frame_count += 1

            # 2. Prediction logic after processing a certain number of frames
            if frame_count % frame_interval == 0:
                keypoints = extract_keypoints(results)
                sequence.append(keypoints)
                sequence = sequence[-20:]

                if len(sequence) == 20:
                    res = model.predict(np.expand_dims(sequence, axis=0))[0]
                    print(actions[np.argmax(res)])
                    predictions.append(np.argmax(res))

                    if np.unique(predictions[-20:])[0]==np.argmax(res):
                        if res[np.argmax(res)] > threshold:

                            if len(sentence) > 0:
                                if actions[np.argmax(res)] != sentence[-1]:
                                    sentence.append(actions[np.argmax(res)])
                            else:
                                sentence.append(actions[np.argmax(res)])

                    if len(sentence) > 5:
                        sentence = sentence[-5:]

                    # Print predicted text
                    predicted_text = ' '.join(sentence)
                    break

        cap.release()
        cv2.destroyAllWindows()
        return(predicted_text)
#make camera
camera = cv2.VideoCapture(1)

def generate_frames():

    while True:
        #get frame
        success, frame = camera.read()

        if not success:
            break
        else:
            #might need to change to png
            result, encoded_image = cv2.imencode('.jpg', frame)
            final = encoded_image.tobytes()

        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')



@app.route('/stream')
def stream():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame') 


###
# Routing for your application.
###
@app.route('/')
def home():
    """Render website's home page."""
    return render_template('home.html')

# @app.route('/translator', methods=['GET', 'POST'])
@app.route('/translator', methods=['GET'])
def translator():

    formSearch = SearchForm()
    formSearch.select_field.choices = getSearchChoices()  
    
    formUpload = UploadForm()
   
    return render_template('translator.html', searchForm=formSearch, uploadForm=formUpload, scrollTo=request.args.get('scrollTo'), translation=request.args.get('translation'))

@app.route('/processUpload', methods=['POST'])
def processUpload():
    form = UploadForm()

    if form.validate_on_submit():
        
        # Get file data and save to your uploads folder
        vid = form.video.data
        filename = secure_filename(vid.filename)
        vid.save(os.path.join(
            app.config['UPLOAD_FOLDER'], filename
        ))
        predicted_text= predict(f'uploads/{filename}')
        
        
        return redirect(url_for('translator', translation=predicted_text, scrollTo='videoUpload'))
    
    for error in form.video.errors:
        flash(error)
   
    return redirect(url_for('translator'))


     

@app.route('/processSearch', methods=['POST'])
def processSearch():
    form = SearchForm()
    form.select_field.choices = getSearchChoices() 

    if form.validate_on_submit():
        
        selected_text = form.select_field.data
        #mak query with selected text

        #query database here
        if selected_text == "What is your Favourite Colour":
            translation = selected_text
        elif selected_text == "How are you":
            translation = selected_text
        elif selected_text == "My name is John":
            translation = selected_text
        elif selected_text == "I am Fine":
            translation = selected_text
        elif selected_text == "My Favourite Colour is Green":
            translation = selected_text
        elif selected_text == "What is your Name":
            translation = selected_text
        elif selected_text == "It is two thirty pm in the evening":
            translation = selected_text
        elif selected_text == "Downtown a few blocks from here":
            translation = selected_text
        elif selected_text == "What time is it":
            translation = selected_text
        elif selected_text == "It is sunny with a chance of clouds":
            translation = selected_text
        elif selected_text == "Yes I could go for a snack":
            translation = selected_text
        elif selected_text == "Are you hungry":
            translation = selected_text
        elif selected_text == "Where is the library":
            translation = selected_text
        elif selected_text == "Do you enjoy sport":
            translation = selected_text
        elif selected_text == "No I dont have any pets":
            translation = selected_text
        elif selected_text == "Economics":
            translation = selected_text
        elif selected_text == "What is your major":
            translation = selected_text
        elif selected_text == "Where is the Library":
            translation = selected_text
        elif selected_text == "Do you have any pets":
            translation = selected_text
        else:
            translation = "Yes I could go for a snack"
            
        return redirect(url_for('translator', translation=translation, scrollTo='search'))
    
    return 'An error occurred with submitting the form'
         


@app.route('/resources')
def resources():
    return render_template('resources.html')


@app.route('/about')
def about():
    return render_template('about.html')

def getSearchChoices():
    # Example: Fetch state choices from database
    # dbFetch = ['Alabama', 'Alaska', 'Alaska']
    # choices = [('', 'Select a state...')]

    # for choice in dbFetch:
    #     choices.append((choice, choice))


     choices = ["What is your Favourite Colour", "How are you", "My Favourite Colour is Green","What is your Name", "My name is John","I am Fine","It is two thirty pm in the evening","Downtown a few blocks from here","Economics",
               "What time is it","It is sunny with a chance of clouds","Yes I could go for a snack","Are you hungry","Where is the library","Do you enjoy sport","No I dont have any pets","What is your major","Do you have any pets"]
     return choices
###
# The functions below should be applicable to all Flask apps.
###

@app.route('/<file_name>.txt')
def send_text_file(file_name):
    """Send your static text file."""
    file_dot_text = file_name + '.txt'
    return app.send_static_file(file_dot_text)


@app.after_request
def add_header(response):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also tell the browser not to cache the rendered page. If we wanted
    to we could change max-age to 600 seconds which would be 10 minutes.
    """
    response.headers['X-UA-Compatible'] = 'IE=Edge,chrome=1'
    response.headers['Cache-Control'] = 'public, max-age=0'
    return response


@app.errorhandler(404)
def page_not_found(error):
    """Custom 404 page."""
    return render_template('404.html'), 404
