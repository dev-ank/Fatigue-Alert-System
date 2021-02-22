import winsound                                       #this module is to run this project locally
import numpy as np
import cv2
import tensorflow as tf

captureDevice = cv2.VideoCapture(0) 

my_model=tf.keras.models.load_model('my_model.h5')

flag=0
while True:
	ret,frame=captureDevice.read()
	eye_cascade=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_eye.xml')
	face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
	gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	
	eyes=eye_cascade.detectMultiScale(gray,1.2,4)

	for x,y,w,h in eyes:
		roi_gray=gray[y:y+h,x:x+w]
		roi_rgb=frame[y:y+h,x:x+w]
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
		eyess=eye_cascade.detectMultiScale(roi_gray)

		if len(eyess)==0:
			print("Eyes not detected")
		else:
			for (ex,ey,ew,eh) in eyess:
				eyes_roi=roi_rgb[ey:ey+eh,ex:ex+ew]


	faces=face_cascade.detectMultiScale(gray,1.3,4)

	for (x,y,w,h) in faces:
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

	font=cv2.FONT_HERSHEY_SIMPLEX

	final_image=cv2.resize(eyes_roi,(224,224))
	final_image=np.expand_dims(final_image,axis=0)
	final_image=final_image/255.0

	prediction=my_model.predict(final_image)

	if (prediction<0.3):
		status="Eyes Open"
		cv2.putText(frame,status,(170,60),font,2,(0,255,0),2,cv2.LINE_4)

		x1,y1,w1,h1=225,435,200,45

		cv2.rectangle(frame,(x1,y1),(x1+w1,y1+h1),(255,0,0),-1)
		cv2.putText(frame,'Driver Active!',(x1+int(w1/10),y1+int(h1/2)),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)
	
	else:
		flag=flag+1
		status="Eyes Closed"
		cv2.putText(frame,status,(170,60),font,2,(0,0,255),2,cv2.LINE_4)
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),0)


		if flag>5:
			x1,y1,w1,h1=225,435,200,45
			cv2.rectangle(frame,(x1,y1),(x1+w1,y1+h1),(0,0,0),-1)
			cv2.putText(frame,'Fatigue Alert!',(x1+int(w1/10),y1+int(h1/2)),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
			winsound.Beep(5000,1000)
			flag=0
			
			
			
	cv2.imshow('Fatigue Alert System', frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

captureDevice.release()
cv2.destroyAllWindows()