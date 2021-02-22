import cv2                                       #this module is for initialization of webcam and getting the frames from the server
from imutils.video import WebcamVideoStream      
import numpy as np
import winsound


class VideoCamera(object):

	def __init__(self,my_model):                     #webcam initialization
		self.stream=WebcamVideoStream(src=0).start()
		self.my_model=my_model

	def __del__(self):                               #stop streaming
		self.stream.stop()

	
	def get_frame(self):                             #fetching the frames from the cloud


		image=self.stream.read()

		eye_cascade=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_eye.xml')
		face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

		face=face_cascade.detectMultiScale(image,1.1,7)           #detects faces
		eyes=eye_cascade.detectMultiScale(image,1.2,4)            #detects eyes
		#gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

		for (x,y,w,h) in eyes:                                    #creates boxes
			#roi_gray=gray[y:y+h,x:x+w]
			roi_rgb=image[y:y+h,x:x+w]
			cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
			eyess=eye_cascade.detectMultiScale(roi_rgb)

			if len(eyess)==0:                   
				print("Eyes not detected")
			else:
				for (ex,ey,ew,eh) in eyess:                        
					eyes_roi=roi_rgb[ey:ey+eh,ex:ex+ew]


				for (x,y,h,w) in face:
					cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)


				font=cv2.FONT_HERSHEY_SIMPLEX

				final_image=cv2.resize(eyes_roi,(224,224))
				final_image=np.expand_dims(final_image,axis=0)
				final_image=final_image/255.0

				prediction=self.my_model.predict(final_image)            #prediction of the current frame

				flag=0

				if (prediction<0.3):                                     #if prob<0.3 eyes open
					status="Eyes Open"
					cv2.putText(image,status,(170,60),font,2,(0,255,0),2,cv2.LINE_4)

					x1,y1,w1,h1=225,435,200,45

					cv2.rectangle(image,(x1,y1),(x1+w1,y1+h1),(255,0,0),-1)
					cv2.putText(image,'Driver Active!',(x1+int(w1/10),y1+int(h1/2)),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)
				
				else:                                                    #prob>0.3 eyes closed...as False Negatives is more crucial in this scenario
					status="Eyes Closed"
					cv2.putText(image,status,(170,60),font,2,(0,0,255),2,cv2.LINE_4)
					cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),0)
					
					
					x1,y1,w1,h1=225,435,200,45
					cv2.rectangle(image,(x1,y1),(x1+w1,y1+h1),(0,0,0),-1)
					cv2.putText(image,'Fatigue Alert!',(x1+int(w1/10),y1+int(h1/2)),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
					winsound.Beep(5000,1000)



		ret,jpeg=cv2.imencode('.jpeg',image)
		data=[]
		data.append(jpeg.tobytes())            #converting the image frame to byte array to send it to server
		return data


