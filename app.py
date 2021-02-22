from webcam import VideoCamera
from flask import Flask,render_template,request,Response
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
import cv2

app=Flask(__name__)                                   #initialization of flask module

my_model=tf.keras.models.load_model('my_model.h5')    #loading the model from disk

@app.route('/')                                        #landing page
def home():
	return render_template('index.html')


def gen(camera):                                       #generator function to generate video streaming on the frontend
	while True:
		data=camera.get_frame()

		frame=data[0]
		
		yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')         #converting to byte array and yeilding frames one at a time

		

@app.route('/video_feed')                            #page to start video streaming
def video_feed():
	return Response(gen(VideoCamera(my_model)),mimetype='multipart/x-mixed-replace; boundary=frame')
	



if __name__=="__main__":
    app.run(debug=False,port=8000)