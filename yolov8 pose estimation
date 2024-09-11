#importing Libraries
from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import time
import openpyxl
import datetime
import os
from openpyxl.styles import Alignment
import io
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import firebase_admin
from firebase_admin import credentials, db
import webbrowser
# import dependencies
from IPython.display import display, Javascript, Image
from google.colab.output import eval_js
from base64 import b64decode, b64encode
import PIL
import html
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from google.colab import drive
from googleapiclient.discovery import build
from google.colab import auth
import threading
from collections import defaultdict

#Importing Webcam in Google Collab
# function to convert the JavaScript object into an OpenCV image
def js_to_image(js_reply):
  """
  Params:
          js_reply: JavaScript object containing image from webcam
  Returns:
          img: OpenCV BGR image
  """
  # decode base64 image
  image_bytes = b64decode(js_reply.split(',')[1])
  # convert bytes to numpy array
  jpg_as_np = np.frombuffer(image_bytes, dtype=np.uint8)
  # decode numpy array into OpenCV BGR image
  img = cv2.imdecode(jpg_as_np, flags=1)

  return img
# JavaScript to properly create our live video stream using our webcam as input
def video_stream():
  js = Javascript('''
    var video;
    var div = null;
    var stream;
    var captureCanvas;
    var imgElement;
    var labelElement;

    var pendingResolve = null;
    var shutdown = false;

    function removeDom() {
       stream.getVideoTracks()[0].stop();
       video.remove();
       div.remove();
       video = null;
       div = null;
       stream = null;
       imgElement = null;
       captureCanvas = null;
       labelElement = null;
    }

    function onAnimationFrame() {
      if (!shutdown) {
        window.requestAnimationFrame(onAnimationFrame);
      }
      if (pendingResolve) {
        var result = "";
        if (!shutdown) {
          captureCanvas.getContext('2d').drawImage(video, 0, 0, 640, 480);
          result = captureCanvas.toDataURL('image/jpeg', 0.8)
        }
        var lp = pendingResolve;
        pendingResolve = null;
        lp(result);
      }
    }

    async function createDom() {
      if (div !== null) {
        return stream;
      }

      div = document.createElement('div');
      div.style.border = '2px solid black';
      div.style.padding = '3px';
      div.style.width = '100%';
      div.style.maxWidth = '600px';
      document.body.appendChild(div);

      const modelOut = document.createElement('div');
      modelOut.innerHTML = "<span>Status:</span>";
      labelElement = document.createElement('span');
      labelElement.innerText = 'No data';
      labelElement.style.fontWeight = 'bold';
      modelOut.appendChild(labelElement);
      div.appendChild(modelOut);

      video = document.createElement('video');
      video.style.display = 'block';
      video.width = div.clientWidth - 6;
      video.setAttribute('playsinline', '');
      video.onclick = () => { shutdown = true; };
      stream = await navigator.mediaDevices.getUserMedia(
          {video: { facingMode: "environment"}});
      div.appendChild(video);

      imgElement = document.createElement('img');
      imgElement.style.position = 'absolute';
      imgElement.style.zIndex = 1;
      imgElement.onclick = () => { shutdown = true; };
      div.appendChild(imgElement);

      const instruction = document.createElement('div');
      instruction.innerHTML =
          '<span style="color: red; font-weight: bold;">' +
          'When finished, click here or on the video to stop this demo</span>';
      div.appendChild(instruction);
      instruction.onclick = () => { shutdown = true; };

      video.srcObject = stream;
      await video.play();

      captureCanvas = document.createElement('canvas');
      captureCanvas.width = 640; //video.videoWidth;
      captureCanvas.height = 480; //video.videoHeight;
      window.requestAnimationFrame(onAnimationFrame);

      return stream;
    }
    async function stream_frame(label, imgData) {
      if (shutdown) {
        removeDom();
        shutdown = false;
        return '';
      }

      var preCreate = Date.now();
      stream = await createDom();

      var preShow = Date.now();
      if (label != "") {
        labelElement.innerHTML = label;
      }

      if (imgData != "") {
        var videoRect = video.getClientRects()[0];
        imgElement.style.top = videoRect.top + "px";
        imgElement.style.left = videoRect.left + "px";
        imgElement.style.width = videoRect.width + "px";
        imgElement.style.height = videoRect.height + "px";
        imgElement.src = imgData;
      }

      var preCapture = Date.now();
      var result = await new Promise(function(resolve, reject) {
        pendingResolve = resolve;
      });
      shutdown = false;

      return {'create': preShow - preCreate,
              'show': preCapture - preShow,
              'capture': Date.now() - preCapture,
              'img': result};
    }
    ''')

  display(js)

def video_frame():
    data = eval_js('stream_frame()')
    return data

#Cheating Detection Algorithm
# start streaming video from webcam
video_stream()


# Authenticate and mount Google Drive
auth.authenticate_user()
drive.mount('/content/drive')

# Path to the credential file on Google Drive
json_file_name = 'fyp-g5-412615-fd33a28e5507.json'
json_path = f'/content/drive/My Drive/{json_file_name}'

# Define the scope
scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']

# Load credentials
creds = ServiceAccountCredentials.from_json_keyfile_name(json_path, scope)

# Authorize the client
client = gspread.authorize(creds)

# Define the folder name
folder_name = 'FYP-G5-UET'

'''
# Find the folder ID using the Google Drive API
drive_service = build('drive', 'v3', credentials=creds)
folder_query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder'"
folders = drive_service.files().list(q=folder_query, fields='files(id)').execute()
folder_id = folders['files'][0]['id']

# Create a new spreadsheet
spreadsheet = client.create('your_spreadsheet_name')

# Move the spreadsheet to the specified folder
file_id = spreadsheet.id
drive_service.files().update(fileId=file_id, addParents=folder_id, fields='id, parents').execute()

# Access the first sheet of the newly created spreadsheet
sheet = spreadsheet.sheet1

# Set headers
sheet.update('A1', 'Time')
sheet.update('B1', 'Message')
sheet.update('C1', 'Image Link')
sheet.update('D1', 'Duration')

print("Google Excel sheet created and initialized in the specified folder.")

'''

# Path to the JSON file on Google Drive
json_file_name = 'fyp-g5-412615-firebase-adminsdk-s6nq4-e74948984e.json'
json_path = f'/content/drive/My Drive/{json_file_name}'

# Load Firebase credentials from the JSON file
cred = firebase_admin.credentials.Certificate(json_path) # Update with your service account key path

dndr = 'auiokuhu0lh'
dn = dndr
# Initialize the Firebase app with a unique name
firebase_admin.initialize_app(cred, name=dn, options={
    'databaseURL': 'https://fyp-g5-412615-default-rtdb.asia-southeast1.firebasedatabase.app/'
})

# Initialize a reference to the Firebase Realtime Database
# Initialize a reference to the Firebase Realtime Database with the specified app name
ref = db.reference('/', app=firebase_admin.get_app(name=dn))

# Define the folder path where the Excel sheet and images are located
folder_path = '/content/drive/My Drive/FYP-2020-G5/'

# Define the base URL for Google Drive file sharing
base_url = 'https://drive.google.com/file/d/'

# Defining Functions
def draw_bb_and_save_image_data(image, text, counter):
    # Get current time
    print("Done")
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Save the image in the same folder as the Excel file
    file_path = os.path.join(folder_path, f"{current_time}.png")
    cv2.imwrite(file_path, image)

    # Generate shareable link for the image
    image_id = file_path.split('/')[-1]
    image_link = base_url + image_id

    # Wrap the data in a dictionary with a suitable key
    firebase_data = {'detection': pose + " for " + str(elapsed_time) + " Seconds and Link is: " + file_path}
    send_to_firebase(firebase_data)
    # Define the path to the Excel file in Google Drive
    excel_file = os.path.join(folder_path, 'f-custom_file_name.xlsx')

    # Check if the Excel file exists, if not, create a new one
    if not os.path.exists(excel_file):
        # Create a new Excel workbook
        wb = openpyxl.Workbook()
        ws = wb.active

        # Set headers
        ws["A1"] = "Time"
        ws["B1"] = "Message"
        ws["C1"] = "Image Link"
        ws["D1"] = "Duration"

        # Save the workbook to the specified path
        wb.save(excel_file)

    # Load the existing workbook
    wb = openpyxl.load_workbook(excel_file)
    ws = wb.active

    # Find the next available row
    next_row = ws.max_row + 1

    # Update the Excel sheet with image information
    ws[f"A{next_row}"] = current_time
    ws[f"B{next_row}"] = text
    ws[f"C{next_row}"].value = '=HYPERLINK("{}","{}")'.format(image_link, "Open Image")
    ws[f"C{next_row}"].style = "Hyperlink"
    ws[f"D{next_row}"] = counter

    # Save the workbook
    wb.save(excel_file)

    print("Image saved and Excel sheet updated.")


# Function to listen for changes in the database
def listen_for_data():

    url = "https://fyp-g5-uet.firebaseapp.com/"
    webbrowser.open(url)

    whatsapp_number = None
    gmail_id = None
    print("Now it is waiting for me to put Whatsapp Number and Gmail ID on webpage")
    while True:
        # Get data from Firebase Realtime Database
        data = ref.get()
        if data:
            for key, value in data['inputs'].items():
                if value.get('type') == 'whatsapp' and whatsapp_number is None:
                    whatsapp_number = value.get('info')
                    print("Received WhatsApp Number:", whatsapp_number)
                elif value.get('type') == 'gmail' and gmail_id is None:
                    gmail_id = value.get('info')
                    print("Received Gmail ID:", gmail_id)

                # Stop listening for data once both WhatsApp number and Gmail ID are received
                if whatsapp_number is not None and gmail_id is not None:
                    ref.child(key).update({'status': 'Functionality executed successfully'})
                    return whatsapp_number, gmail_id

                # After executing functionality, you can update the HTML page or send information back to the database
                # For example:
            time.sleep(1)  # Adjust delay as needed

# Call the function to listen for data changes and retrieve WhatsApp number and Gmail ID
whatsapp_number, gmail_id = listen_for_data()

# Now you can use whatsapp_number and gmail_id wherever needed in your code
print("WhatsApp Number:", whatsapp_number)
print("Gmail ID:", gmail_id)
print("Now it will run algorithms and it will send detections to webpage")

def send_email(sender_email, sender_password, recipient_email, subject, body, smtp_server='smtp.gmail.com', smtp_port=587):
    try:
        # Set up the SMTP server
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(sender_email, sender_password)

        # Create a message
        message = MIMEMultipart()
        message['From'] = sender_email
        message['To'] = recipient_email
        message['Subject'] = subject

        # Add the body of the email
        message.attach(MIMEText(body, 'plain'))

        # Send the email
        server.sendmail(sender_email, recipient_email, message.as_string())
        print("Email sent successfully!")

        # Close the connection
        server.quit()

    except Exception as e:
        print(f"Failed to send email. Error: {e}")

send_email("sajjadhaider1816@gmail.com", "srjj ncof pmoe zqhi" ,"haidersajjadseo@gmail.com","Cheating Detection", "https://drive.google.com/drive/folders/14oBp49vK2ThNOCR3X6U2i-M9-PFsYkMa?usp=sharing" , smtp_server='smtp.gmail.com', smtp_port=587)
# Function to send data to Firebase
def send_to_firebase(data):
    # Initialize a reference to the Firebase Realtime Database with the specified app name
    ref = db.reference('/', app=firebase_admin.get_app(name=dn))

    # Push data to Firebase
    ref.push(data)

    try:
        # Update the Firebase database with the provided data
        ref.update(data)
        print("Data sent to Firebase successfully.")
    except Exception as e:
        print(f"Failed to send data to Firebase: {e}")

def calculate_angle(p1, p2, p3):
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    angle_rad = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    angle_deg = np.degrees(angle_rad)
    return angle_deg

def estimate_pose_both_eyes_visible(nose, left_eye, right_eye, left_ear, right_ear, nose_n):
    point_on_line = [0.5,0]
    camera = [0.5, 0.5]
    camera_tensor = torch.tensor(camera)
    point_on_line_tensor = torch.tensor(point_on_line)
    # Calculate angles for both eyes visible
    angle1 = calculate_angle(nose, left_eye, right_eye)
    angle2 = calculate_angle(left_eye, right_eye, nose)
    angle3 = calculate_angle(right_eye, nose, left_eye)
    angle4 = calculate_angle(nose, left_ear, right_ear)
    angle5 = calculate_angle(left_ear, right_ear, nose)
    relative_angle = calculate_angle(nose_n, point_on_line_tensor, camera_tensor)
    print("Left_eye:",angle1,", Right Eye: ", angle2,", Nose: ", angle3, ", Relative Angle: ",relative_angle)

    # Base Threshold
    base_Threshold = 60

    # Define thresholds for looking left, looking right, and looking straight
    if nose_n[0] > 0.5:
        left_threshold = base_Threshold + relative_angle/2.2
        right_threshold = base_Threshold - relative_angle/2.2
    else:
        left_threshold = base_Threshold - relative_angle/2.2
        right_threshold = base_Threshold + relative_angle/2.2

    # Print Threshold
    print("Left Threshold: ", left_threshold, ", Right Threshold: ", right_threshold)

    #Check if person is looking backword
    if int(relative_angle) == 90:
        return "Looking Backward"

    # Check if any eye is invisible
    elif angle1 < 5 or angle2 < 5:
        if angle1 < 5:
            if angle4 < 5:
                return "Looking Left"
            else:
                if angle4 > left_threshold:
                    return "Looking Left"
                elif angle5 > right_threshold:
                    return "Looking Right"
        else:
            if angle5 < 5:
                return "Looking Right"
            else:
                if angle4 > left_threshold:
                    return "Looking Left"
                elif angle5 > right_threshold:
                    return "Looking Right"

    # Check if both angles cross threshold
    elif angle1 > left_threshold and angle2 > right_threshold:
        if angle1 > angle2:
            return "Looking Left"
        else:
            return "Looking Right"

    # Check if the person is looking left
    elif angle1 > left_threshold:
        if relative_angle <= 10 and (angle4 < 5 or angle5 < 5):
            if angle4 < 5:
                return "Looking Left"
            if angle5 < 5:
                return "Looking Right"
        else:
            return "Looking Left"

    # Check if the person is looking right
    elif angle2 > right_threshold:
        if relative_angle <= 10 and (angle4 < 5 or angle5 < 5):
            if angle4 < 5:
                return "Looking Left"
            if angle5 < 5:
                return "Looking Right"
        else:
            return "Looking Right"

    # If not looking left or right, consider as looking straight
    else:
        if relative_angle <= 10 and (angle4 < 5 or angle5 < 5):
            if angle4 < 5:
                return "Looking Left"
            if angle5 < 5:
                return "Looking Right"
        else:
            return "Looking Straight"

# Function to save image and update Excel sheet
def save_image_and_update_excel(image, pose, elapsed_time):
    # Get current time
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Save the image
    image_filename = f"{current_time}.png"
    image_path = os.path.join("/content/drive/My Drive/FYP-G5-UET/", image_filename)
    cv2.imwrite(image_path, image)


    # Update Excel sheet
    update_excel_sheet(current_time, pose, image_filename, elapsed_time)

# Function to update Excel sheet
def update_excel_sheet(current_time, pose, image_path, elapsed_time):
    # Define the path to the Excel file in Google Drive
    excel_file_path = "/content/drive/My Drive/FYP-G5-UET/fyp-2020.xlsx"

    # Load the existing workbook or create a new one if it doesn't exist
    if os.path.exists(excel_file_path):
        workbook = openpyxl.load_workbook(excel_file_path)
    else:
        workbook = openpyxl.Workbook()
        workbook.active.append(['Time', 'Message', 'Image Link', 'Duration'])

    # Get the active sheet
    sheet = workbook.active

    # Append new row with data
    sheet.append([current_time, pose, image_path, elapsed_time])

    # Save the workbook
    workbook.save(excel_file_path)

    print("Image saved and Excel sheet updated.")

# Load a model
model = YOLO('yolov8n-pose.pt')  # pretrained YOLOv8n model

# Define a dictionary to track the start time for each face
face_start_time = {}

while True:
    js_reply = video_frame()

    # convert JS response to OpenCV Image
    frame = js_to_image(js_reply["img"])

    # Perform object detection
    results = model.track(frame, conf=0.25)
    track_ids = results[0].boxes.id.int().cpu().tolist()
    # Access the keypoints
    keypoints = results[0].keypoints
    data = keypoints.xy
    data1 = keypoints.xyn

    # Define a dictionary to track each detection's track ID and associated information
    detection_info = {}
    detections = []
    # Iterate over each detected pose
    for i in range(len(data)):
        nose = data[i][0]
        left_eye = data[i][1]
        right_eye = data[i][2]
        left_ear = data[i][3]
        right_ear = data[i][4]
        nose_n = data1[i][0]

        # Estimate pose
        pose = estimate_pose_both_eyes_visible(nose, left_eye, right_eye, left_ear, right_ear, nose_n)
        print("Pose: ", pose)
        # Get the track ID
        track_id = track_ids[i]

        # Check if the track ID already exists in the dictionary
        if track_id not in detection_info:
            detection_info[track_id] = {'elapsed_time': None, 'pose': None, 'counter':0}

        # Draw bounding box if pose meets criteria
        if pose in ["Looking Left", "Looking Right", "Looking Backward"]:
            if face_start_time.get(i) is None:
                face_start_time[i] = time.time()
                detection_info[track_id]['counter'] = 0
            else:
                elapsed_time = time.time() - face_start_time[i]
                print(elapsed_time, " > ", 5 + detection_info[track_id]['counter'])
                if elapsed_time >= 5 + detection_info[track_id]['counter']:
                    detection_info[track_id]['counter'] += 3
                    # Draw bounding box
                    if left_eye.tolist()[0] > 0 and right_eye.tolist()[0] > 0 and left_ear.tolist()[0] > 0 and right_ear.tolist()[0] > 0:
                        x1 = int(min(left_eye.tolist()[0], right_eye.tolist()[0], right_ear.tolist()[0], left_ear.tolist()[0]) / 1.01)
                        x2 = int(max(left_eye.tolist()[0], right_eye.tolist()[0], right_ear.tolist()[0], left_ear.tolist()[0]) * 1.01)
                        y1 = int(min(left_eye.tolist()[1], right_eye.tolist()[1], right_ear.tolist()[1], left_ear.tolist()[1]) / 1.02)
                        y2 = int(max(left_eye.tolist()[1], right_eye.tolist()[1], right_ear.tolist()[1], left_ear.tolist()[1]) * 1.025)
                    else:
                        x1 = int(nose.tolist()[0] / 1.02)
                        x2 = int(nose.tolist()[0] * 1.02)
                        y1 = int(nose.tolist()[1] / 1.02)
                        y2 = int(nose.tolist()[1] * 1.02)

                    # Update detection info
                    detection_info[track_id]['pose'] = pose
                    detection_info[track_id]['elapsed_time'] = elapsed_time
                    detections.append((x1,y1,x2,y2))
                else:
                  detection_info[track_id]['elapsed_time'] = None
        else:
            detection_info[track_id] = {'elapsed_time': None, 'pose': None}
            face_start_time[i] = None
        # Draw bounding boxes on the image
    for (x1, y1, x2, y2) in detections:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green color, thickness 2
        print(detections)


    # Call the function to save image and update Excel for each detection
    for track_id, info in detection_info.items():
        if info['elapsed_time'] is not None:
            save_image_and_update_excel(frame, info['pose'], info['elapsed_time'])
            str = "Person is ", info['pose'], " for ", info['elapsed_time'], " seconds"
            data = {'detection':str}
            send_to_firebase(data)
