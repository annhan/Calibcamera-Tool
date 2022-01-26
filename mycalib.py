#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import cv2
import numpy as np
import yaml
#from scipy import optimize
from PyQt5.QtWidgets import  QApplication, QMainWindow,  QMessageBox , QWidget
from PyQt5 import uic
from PyQt5.QtCore import Qt,QTimer ,pyqtSignal,QRect
from PyQt5.QtGui import QImage, QPixmap, QPainter,QDoubleValidator
import time
import Object_detect

sys.stdout.flush()
#Config Variables - Enter their values according to your Checkerboard
"""
F1,
indentationToSpaces or indentationToTabs (depending on your need)
Enter.
"""
class StateVari(QWidget):
	valueChanged = pyqtSignal(object)

	def __init__(self, parent=None):
		super(StateVari, self).__init__(parent)
		self._t = 0

	@property
	def t(self):
		return self._t

	@t.setter
	def t(self, value):
		self._t = value
		self.valueChanged.emit(value)

class camera_realtimeXYZ:

    #camera variables
    cam_mtx=None
    dist=None
    newcam_mtx=None
    roi=None
    rvec1=None
    tvec1=None
    R_mtx=None
    Rt=None
    P_mtx=None

    #images
    img=None

    def __init__(self):

        imgdir="/home/pi/Desktop/Captures/"
        savedir="camera_data/"
        self.imageRec=Object_detect.image_recognition(True,False,imgdir,imgdir,False,False,True)

        #self.imageRec=image_recognition_singlecam.image_recognition(True,False,imgdir,imgdir,True,True)
        self.cam_mtx=None
        self.dist=None
        self.newcam_mtx=None
        self.roi=None
        self.rvec1=None
        self.tvec1=None
        self.R_mtx=None
        self.Rt=None
        self.P_mtx=None
        s_arr=[0]
        try:
            s_arr=np.load('./output/s_arr.npy')
        except:
            s_arr=[0]
        self.scalingfactor=s_arr[0]
        self.inverse_newcam_mtx = None
        self.inverse_R_mtx = None
        self.backg = None
        #self.inverse_newcam_mtx = np.linalg.inv(self.newcam_mtx)
        #self.inverse_R_mtx = np.linalg.inv(self.R_mtx)
    def updateCamMtx(self,cam_mtx,dist,new_camera_matrix,roi):
        self.cam_mtx=cam_mtx
        self.dist=dist
        self.newcam_mtx=new_camera_matrix
        self.roi=roi

    def updatePara(self,rvec1,tvec1,R_mtx,Rt,P_mtx):
        self.rvec1=rvec1
        self.tvec1=tvec1
        self.R_mtx=R_mtx
        self.Rt=Rt
        self.P_mtx=P_mtx
        #self.scalingfactor=s_arr[0]
        self.inverse_newcam_mtx = np.linalg.inv(self.newcam_mtx)
        self.inverse_R_mtx = np.linalg.inv(self.R_mtx)

    def updateArr(self,s_arrIN):
        s_arr=s_arrIN
        self.scalingfactor=s_arr[0]

    def previewImage(self, text, img):
        #show full screen
        cv2.namedWindow(text, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(text,cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

        cv2.imshow(text,img)
        cv2.waitKey(2000)
        cv2.destroyAllWindows()

    def undistort_image(self,image):
        image_undst = cv2.undistort(image, self.cam_mtx, self.dist, None, self.newcam_mtx)
        x, y, w, h = self.roi
        image_undst = image_undst[y:y + h, x:x + w]
        image_undst = cv2.resize(image_undst, (w, h))
        return image_undst

    def load_background(self,background):
        bg_undst=self.undistort_image(background)
        self.backg = bg_undst

    def detect_xyz(self,image,calcXYZ=True,calcarea=False):

        image_src=image.copy()
        
        img=image_src
        bg=self.backg
                    
        XYZ=[]
        obj_count, detected_points, img_output=self.imageRec.run_detection(img,bg)

        if (obj_count>0):

            for i in range(0,obj_count):
                x=detected_points[i][0]
                y=detected_points[i][1]
                w=detected_points[i][2]
                h=detected_points[i][3]
                cx=detected_points[i][4]
                cy=detected_points[i][5]
                angle = detected_points[i][6]
                if calcXYZ==True:
                    XYZ.append(self.calculate_XYZ(cx,cy))
                    inname = "X,Y: "+str(self.truncate(XYZ[i][0],2))+","+str(self.truncate(XYZ[i][1],2))
                    print("inname" , inname , flush=True)
                    cv2.putText(img_output,inname,(int(x),int(y+60)),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,255,0),1, cv2.LINE_AA)
                #if calcarea==True:
                #    cv2.putText(img_output,"area: "+str(self.truncate(w*h,2)),(x,y-12),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)

        return img_output, XYZ

    def calculate_XYZ(self,u,v):
                                      
        #Solve: From Image Pixels, find World Points
        
        uv_1=np.array([[u,v,1]], dtype=np.float32)
        uv_1=uv_1.T
        suv_1=self.scalingfactor*uv_1
        xyz_c=self.inverse_newcam_mtx.dot(suv_1)
        xyz_c=xyz_c-self.tvec1
        XYZ=self.inverse_R_mtx.dot(xyz_c)

        return XYZ


    def truncate(self, n, decimals=0):
        n=float(n)
        multiplier = 10 ** decimals
        return int(n * multiplier) / multiplier

class mycalib(QMainWindow):


	def __init__(self):
		super(mycalib, self).__init__()
		uic.loadUi('mycLIB.ui', self)
		self.show()
		self.id_counter = 0
		self.obj_detected=0
		self.obj_detected_prev=0
		self.objls = [] # 3d point in real world space
		self.imgls = [] # 2d points in image plane.
		self.tot_error = 0
		self.n_col = 9   #number of columns of your Checkerboard
		self.n_row = 7  #number of rows of your Checkerboard
		self.square_size = 21.0 # size of square on the Checkerboard in mm
		self.pixmap = None
		self.capturing = StateVari()
		self.capturing.valueChanged.connect(self.capturing_change)
		self.capturing.t = 0
		self.confirmedImagesCounter = 0 # how many images are confirmed by user so far
		self.detectingCorners = False
		self.currentCorners = None # array of last detected corners
		self.currentcornerSubPixs = None
		self.predicted2D = None

		self.matrix = np.zeros((3, 3), np.float)   #extracting camera_matrix key and convert it into Numpy Array (2D Matrix)
		self.dist = np.zeros((1, 5))
		self.new_camera_matrix = None
		self.roi = None
		self.diffAng = 0

		self.timer = QTimer(self, interval=50, timeout=self.handle_timeout)
		self.timer.start(50)

		#self.imageRec=Object_detect.image_recognition(False,False,"","",False,True,True)
		camnum = 1
		for i in range(1,10):
			cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
			if(cap.isOpened()):
				camnum = i
				cap.release()
				break
			cap.release()
		print("CAM NUM ",camnum)
		#cam = cv2.VideoCapture(camnum)
		self.cap = cv2.VideoCapture(camnum, cv2.CAP_DSHOW) # webcam object
		self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
		self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
		self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) 
		self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
		self.cameraXYZ=camera_realtimeXYZ()
		self.bg = None
		self.load_paramsyaml()
		self.loadAllpara()
		ret,frame = self.cap.read()
		if (ret): self.bg = self.cameraXYZ.undistort_image(frame)
		self.initEvents() 
		self.initUi()
	
	def initUi(self):
		axes =   ['1X','1Y','2X','2Y','3X','3Y','4X','4Y','5X','5Y','6X','6Y','7X','7Y']
		valueR = [0   , 0  , 147, 0  , 147, 189,   0, 189,   0, 105, 147, 105,  84, 105]
		valuaP = [98.5,104.5,306.3,39.1,441.5,309.2,221.5,414.5,164.5,289.2,380,185.5,287.4,230.8]
		for axis in axes: 
			getattr(self, 'lbl_Point' + axis).setValidator(QDoubleValidator())
			getattr(self, 'lbl_RPoint' + axis).setValidator(QDoubleValidator())
			self.setTextToGui(getattr(self, 'lbl_Point' + axis),str(valuaP[axes.index(axis)]),None)
			self.setTextToGui(getattr(self, 'lbl_RPoint' + axis),str(valueR[axes.index(axis)]),None)
		self.setTextToGui(self.lbl_RAngle,"0",None)
		self.setTextToGui(self.lbl_OAngle,"64",None)


	def setTextToGui(self,name,text = None,color = None):
		if (text != None):
			name.setText(text)
		if (color != None):
			name.setStyleSheet("background-color : {}".format(color))

	def capturing_change(self):
		if (self.capturing.t == 1):
			self.setTextToGui(self.captureButton,"Running","green")
			self.setTextToGui(self.btn_detectpoints,"Detect Points","red")
			self.setTextToGui(self.btn_runRealWorld,"Run Real World detect","red")
		elif (self.capturing.t == 2):
			self.setTextToGui(self.captureButton,"START","red")
			self.setTextToGui(self.btn_detectpoints,"Running","green")
			self.setTextToGui(self.btn_runRealWorld,"Run Real World detect","red")
		elif (self.capturing.t == 3):
			self.setTextToGui(self.captureButton,"START","red")
			self.setTextToGui(self.btn_detectpoints,"Detect Points","red")
			self.setTextToGui(self.btn_runRealWorld,"Running","green")
		else:
			self.setTextToGui(self.captureButton,"START","red")
			self.setTextToGui(self.btn_detectpoints,"Detect Points","red")
			self.setTextToGui(self.btn_runRealWorld,"Run Real World detect","red")

	def handle_timeout(self):
		self.timer.stop()
		self.update()
		
	def getTimeBlock(self):
		timebegin = time.time()
		return time.time() - timebegin

	def initEvents(self):
		""" initialize click events (listeners) """
		self.captureButton.clicked.connect(self.btn_capture_Clicked)
		self.ignoreButton.clicked.connect(self.btn_ignore_Clicked)
		self.confirmButton.clicked.connect(self.btn_confirm_Clicked)
		self.doneButton.clicked.connect(self.btn_done_Clicked)
		self.detectCornersButton.clicked.connect(self.btn_detectCorners_Clicked)
		self.btn_detectpoints.clicked.connect(self.btn_detectpoints_Clicked)
		self.btn_calibpoints.clicked.connect(self.btn_calibPoints_Clicked)
		self.btn_addbg.clicked.connect(self.btn_addbg_Clicked)
		self.btn_runRealWorld.clicked.connect(self.btn_runRealWorld_Clicked)
		self.btn_cacRealWorld.clicked.connect(self.btn_cacRealWorld_Clicked)

		
	def load_paramsyaml(self, param_file:str='./output/calibration.yaml'):
		self.matrix=np.load('./output/cam_mtx.npy')
		self.dist=np.load('./output/dist.npy')
		self.new_camera_matrix = np.load('./output/inverse_newcam_mtx.npy' )
		self.roi = np.load('./output/roi.npy' )
		self.cameraXYZ.updateCamMtx(self.matrix,self.dist,self.new_camera_matrix,self.roi)

	def loadAllpara(self):
		self.matrix = np.load('./output/cam_mtx.npy' )
		self.dist = np.load('./output/dist.npy')
		self.new_camera_matrix = np.load('./output/inverse_newcam_mtx.npy' )
		self.roi = np.load('./output/roi.npy' )
		rvec1 = np.load('./output/rvec1.npy' )
		tvec1 = np.load('./output/tvec1.npy' )
		R_mtx = np.load('./output/R_mtx.npy' )
		Rt = np.load('./output/Rt.npy' )
		P_mtx = np.load('./output/P_mtx.npy' )
		s_arr = np.load('./output/s_arr.npy' )
		self.cameraXYZ.updateCamMtx(self.matrix,self.dist,self.new_camera_matrix,self.roi)
		self.cameraXYZ.updatePara(rvec1,tvec1,R_mtx,Rt,P_mtx)
		self.cameraXYZ.updateArr(s_arr) 

	def runTest(self):
		ret, frame = self.captureCamera()
		if ret:
			self.obj_detected_prev=self.obj_detected
			frame=self.cameraXYZ.undistort_image(frame)
			image, XYZ = self.cameraXYZ.detect_xyz(frame,calcXYZ=True)
			self.obj_detected=len(XYZ)
			if self.obj_detected>0 and self.obj_detected==self.obj_detected_prev:
				self.id_counter = self.id_counter  + 1
			if self.id_counter>20:
				cv2.putText(image,"Picking",(20,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
				print("Trigger Arm")
				self.id_counter=0
			self.pixmap = self.imageToPixmap(image)
			#run detection when on

	def captureRealImage(self):
		ret, frame = self.captureCamera()
		if ret:
			frame = self.cameraXYZ.undistort_image(frame)
			obj_count, detected_points, img_output=self.cameraXYZ.imageRec.run_detection(frame,self.bg)
			#img_output = cv2.circle(img_output,(320,240), 5, (0,0,255), -1)
			#img_output = cv2.circle(img_output,(220,180), 5, (0,0,255), -1)
			#img_output = cv2.circle(img_output,(320,180), 5, (0,0,255), -1)
			#img_output = cv2.circle(img_output,(420,180), 5, (0,0,255), -1)
			#img_output = cv2.circle(img_output,(220,300), 5, (0,0,255), -1)
			#img_output = cv2.circle(img_output,(320,300), 5, (0,0,255), -1)
			#img_output = cv2.circle(img_output,(420,300), 5, (0,0,255), -1)
			self.pixmap = self.imageToPixmap(img_output)

	def paintEvent(self, event):
		painter = QPainter(self)
		if self.pixmap: # display image taken from webcam
			self.imageLabel.setAlignment(Qt.AlignLeft|Qt.AlignTop)
			self.imageLabel.setText('')
			rect = QRect(280, 10, self.width, self.height)
			painter.drawPixmap(rect, self.pixmap)
			#painter.drawPixmap(10, 60, self.pixmap)
		if (self.capturing.t == 1):self.captureImage()
		elif (self.capturing.t == 2):self.captureRealImage()
		elif (self.capturing.t == 3):self.runTest()
		self.timer.start(10)

	def btn_capture_Clicked(self):
		if self.capturing.t != 1:
			self.capturing.t = 1
		elif self.capturing.t == 1:
			self.capturing.t = 0

	def captureCamera(self):
		return self.cap.read() # read frame from webcam

	def captureImage(self):
		ret, frame = self.captureCamera() # read frame from webcam
		if ret: # if frame captured successfully
			#frame_inverted = cv2.flip(frame, 1) # flip frame horizontally
			frame_inverted = frame
			if self.detectingCorners: # if detect corners checkbox is checked
				cornersDetected, corners, imageWithCorners , cornerSubPixs = self.detectCorners(frame_inverted) # detect corners on chess board
				if cornersDetected: # if corners detected successfully
					self.currentCorners = corners
					self.currentcornerSubPixs = cornerSubPixs
					self.frameWithCornersCaptured()
					self.detectingCorners = False
					self.setTextToGui(self.detectCornersButton,'Detect Corners',"red")
			self.pixmap = self.imageToPixmap(frame_inverted)

	def detectCorners(self, image):
		criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		ret, corners = cv2.findChessboardCorners(gray, (self.n_row,self.n_col ), cv2.CALIB_CB_FAST_CHECK)
		cornerSubPixs = None
		if ret:
			cornerSubPixs = cv2.cornerSubPix(gray, corners, (10, 10), (-1,-1), criteria)
			cv2.drawChessboardCorners(image, (self.n_row,self.n_col ), corners, ret)
		return ret, corners, image , cornerSubPixs

	def frameWithCornersCaptured(self):
		self.btn_capture_Clicked() #fix last frame
		self.toggleConfirmAndIgnoreVisibility(True)

	def cal_real_corner(self, corner_height, corner_width, square_size):
		obj_corner = np.zeros([self.n_row * self.n_col, 3], np.float32)
		obj_corner[:, :2] = np.mgrid[0:self.n_row, 0:self.n_col].T.reshape(-1, 2)  # (w*h)*2
		#[0:n_row,0:n_col]
		return obj_corner * square_size

	def mycalibChess(self):
		ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.objls, self.imgls, (self.width,self.height), None, None)
		if not ret:
			print("Cannot compute calibration!",flush=True)		
		else:
			print("Camera calibration successfully computed",flush=True)
			self.new_camera_matrix, self.roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(self.width,self.height),1,(self.width,self.height))
			# Compute reprojection errors
			for i in range(len(self.objls)):
				imgpoints2, _ = cv2.projectPoints(self.objls[i], rvecs[i], tvecs[i], mtx, dist)
				error = cv2.norm(self.imgls[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
				self.tot_error += error
			print("Camera matrix: ", mtx,flush=True)
			print("Distortion coeffs: ", dist,flush=True)
			print("Total error: ", self.tot_error,flush=True)
			print("Mean error: ", np.mean(error),flush=True)		
			# Saving calibration matrix
			np.save('./output/cam_mtx.npy', mtx)
			np.save('./output/dist.npy', dist)
			np.save('./output/inverse_newcam_mtx.npy', self.new_camera_matrix)
			np.save('./output/roi.npy', self.roi)

			result_file = "./output/calibration.yaml"
			print("Saving camera matrix .. in ",result_file,flush=True)
			data={"camera_matrix": mtx.tolist(), "dist_coeff": dist.tolist(), "new_camera_matrix": self.new_camera_matrix, "roi": self.roi}
			with open(result_file, "w") as f:
				yaml.dump(data, f, default_flow_style=False)


	
	def toggleConfirmAndIgnoreVisibility(self, visibility=True):
		if visibility:
			self.ignoreButton.show()
			self.confirmButton.show()
		else:
			self.ignoreButton.hide()
			self.confirmButton.hide()

	def imageToPixmap(self, image):
		qformat = QImage.Format_RGB888
		img = QImage(image, image.shape[1], image.shape[0] , image.strides[0], qformat)
		img = img.rgbSwapped()  # BGR > RGB
		return QPixmap.fromImage(img)

	def displayImage(self):
		self.imageLabel.setPixmap(self.pixmap)

########################################################################
### Event Buttons Click ################################################
#########################################################################

	def btn_detectpoints_Clicked(self):
			#self.load_paramsyaml()
			self.loadAllpara()
			self.btn_addbg_Clicked()
			if (self.capturing.t != 2):
				self.capturing.t = 2
			elif (self.capturing.t == 2):
				self.capturing.t = 0

	def btn_cacRealWorld_Clicked(self):
		self.loadAllpara()
		pointX,pointY,Angle = float(self.lbl_calPointX.text()) , float(self.lbl_calPointY.text()), float(self.lbl_calAngle.text())
		print("Caculator",pointX,pointY,flush=True)
		XYZ = self.cameraXYZ.calculate_XYZ(pointX,pointY)
		AngleNew = Angle - self.diffAng
		print("XYZ",XYZ,flush=True)
		X = self.cameraXYZ.truncate(XYZ[0],2)
		Y = self.cameraXYZ.truncate(XYZ[1],2)
		self.setTextToGui(self.lbl_calResY,text=str(Y))
		self.setTextToGui(self.lbl_calResX,text=str(X))
		self.setTextToGui(self.lbl_calResAngle,text=str(AngleNew))

	def btn_runRealWorld_Clicked(self):
		self.loadAllpara()
		ret, frame = self.captureCamera()
		if ret:
			self.cameraXYZ.load_background(frame)

		if (self.capturing.t != 3):
			self.capturing.t = 3
		elif (self.capturing.t == 3):
			self.capturing.t = 0


	def btn_addbg_Clicked(self):
		ret,frame = self.captureCamera()
		if (ret): 
			self.bg = self.cameraXYZ.undistort_image(frame)
			self.cameraXYZ.load_background(frame)


	def btn_confirm_Clicked(self):
		self.confirmedImagesCounter += 1
		obj_corner = self.cal_real_corner(self.n_row,self.n_col, self.square_size)
		self.objls.append(obj_corner)
		self.imgls.append(self.currentcornerSubPixs)
		self.toggleConfirmAndIgnoreVisibility(False)
		self.btn_capture_Clicked() #continue capturing
		self.setTextToGui(self.lbl_numpics,text=str(self.confirmedImagesCounter))

	def btn_ignore_Clicked(self):
		self.btn_capture_Clicked() #continue capturing
		self.toggleConfirmAndIgnoreVisibility(False)

	def btn_done_Clicked(self):
		if self.confirmedImagesCounter < 10:
			rem = 10 - self.confirmedImagesCounter
			QMessageBox.question(self, 'Warning!', "the number of captured photos should be at least 10. Please take "+str(rem)+" more photos",QMessageBox.Ok)
		else:		
			self.btn_capture_Clicked() #stop capturing
			self.mycalibChess()
			self.objls = [] # 3d point in real world space
			self.imgls = [] # 2d points in image plane.
			self.confirmedImagesCounter = 0
			self.setTextToGui(self.lbl_numpics,text=str(self.confirmedImagesCounter))

	def btn_detectCorners_Clicked(self):
		self.detectingCorners = True
		self.setTextToGui(self.detectCornersButton,None,"green")
		self.setTextToGui(self.captureButton,"Waiting Corner",None)

	def btn_calibPoints_Clicked(self):
		self.load_paramsyaml()
		total_points_used=7
		worldPoints=  np.array([[250,250,100],
								[50,50,100],
								[250,50,100],
								[450,50,100],
								[50,450,100],
								[250,450,100],
								[450,450,100]], dtype=np.float32)
		

		imagePoints=  np.array([[320,240],
								[220,140],
								[320,140],
								[420,140],
								[220,340],
								[320,340],
								[420,340]], dtype=np.float32)
		labels = ['1X','2X','3X','4X','5X','6X','7X','1Y','2Y','3Y','4Y','5Y','6Y','7Y']
		number =[0,0,1,0,2,0,3,0,4,0,5,0,6,0,0,1,1,1,2,1,3,1,4,1,5,1,6,1]
		for label in labels: # add each axis position currentIndex
			index = labels.index(label) * 2
			indext = index + 1
			worldPoints[number[index],number[indext]]=float(getattr(self, 'lbl_RPoint' + label).text())
			imagePoints[number[index],number[indext]]=float(getattr(self, 'lbl_Point' + label).text())
			print("imagePoints",imagePoints[number[index],number[indext]],"worldPoints",worldPoints[number[index],number[indext]],flush=True)		
		#	
		"""for i in range(1,total_points_used):
			#start from 1, given for center Z=d*
			#to center of camera
			wX=worldPoints[i,0]-worldPoints[0,0]
			wY=worldPoints[i,1]-worldPoints[0,1]
			wd=worldPoints[i,2]

			d1=np.sqrt(np.square(wX)+np.square(wY))
			wZ=np.sqrt(np.square(wd)-np.square(d1))
			worldPoints[i,2]=wZ"""

		mtx = self.matrix
		dist = self.dist
		w = self.width
		h =	self.height
		new_camera_matrix=self.new_camera_matrix
		roi=self.roi
		inverse_newcam_mtx = np.linalg.inv(new_camera_matrix)
		ret, rvec1, tvec1=cv2.solvePnP(worldPoints,imagePoints,new_camera_matrix,dist)
		R_mtx, jac=cv2.Rodrigues(rvec1)
		Rt=np.column_stack((R_mtx,tvec1))
		P_mtx=new_camera_matrix.dot(Rt)
		s_arr=np.array([0], dtype=np.float32)
		s_describe=np.array([0,0,0,0,0,0,0,0,0,0],dtype=np.float32)
		self.cameraXYZ.updatePara(rvec1,tvec1,R_mtx,Rt,P_mtx)
		for i in range(0,total_points_used):
			print("=======POINT # " + str(i) +" =========================",flush=True)
			#print("Forward: From World Points, Find Image Pixel")
			XYZ1=np.array([[worldPoints[i,0],worldPoints[i,1],worldPoints[i,2],1]], dtype=np.float32)
			XYZ1=XYZ1.T
			print("WORLD INPUT",flush=True)
			print(XYZ1,flush=True)
			suv1=P_mtx.dot(XYZ1)
			#print("//-- suv1",flush=True)
			#print(suv1)
			s=suv1[2,0]    
			uv1=suv1/s
			print(">==> uv1 - Image Points",flush=True)
			print(uv1)
			#print(">==> s - Scaling Factor",flush=True)
			#print(s)
			s_arr=np.array([s/total_points_used+s_arr[0]], dtype=np.float32)
			s_describe[i]=s
			#if writeValues==True: np.save(savedir+'s_arr.npy', s_arr)
			#print("Solve: From Image Pixels, find World Points",flush=True)
			uv_1=np.array([[imagePoints[i,0],imagePoints[i,1],1]], dtype=np.float32)
			uv_1=uv_1.T
			#print(">==> uv1")
			#print(uv_1)
			suv_1=s*uv_1
			#print("//-- suv1")
			#print(suv_1)
			#print("get camera coordinates, multiply by inverse Camera Matrix, subtract tvec1",flush=True)
			xyz_c=inverse_newcam_mtx.dot(suv_1)
			xyz_c=xyz_c-tvec1
			#print(" xyz_c",flush=True)
			inverse_R_mtx = np.linalg.inv(R_mtx)
			XYZ=inverse_R_mtx.dot(xyz_c)
			print("XYZ",flush=True)
			print(XYZ,flush=True)
			cXYZ=self.cameraXYZ.calculate_XYZ(imagePoints[i,0],imagePoints[i,1])
			print("WORLD OUT ",flush=True)
			print(cXYZ,flush=True)

		print("s_arr",s_arr,flush=True)
		self.cameraXYZ.updatePara(rvec1,tvec1,R_mtx,Rt,P_mtx)
		self.cameraXYZ.updateArr(s_arr)

		angRobot = float(self.lbl_RAngle.text())
		angObj = float(self.lbl_OAngle.text())
		self.diffAng = angObj  - angRobot
		np.save('./output/rvec1.npy', rvec1)
		np.save('./output/tvec1.npy', tvec1)
		np.save('./output/R_mtx.npy', R_mtx)
		np.save('./output/Rt.npy', Rt)
		np.save('./output/P_mtx.npy', P_mtx)
		np.save('./output/s_arr.npy', s_arr)

		result_file = "./output/realCamera.yaml"
		data={"rvec1": rvec1,"tvec1": tvec1, "R_mtx": R_mtx, "Rt": Rt, "P_mtx": P_mtx, "s_arr": s_arr}
		with open(result_file, "w") as f:
			yaml.dump(data, f, default_flow_style=False)
		s_mean, s_std = np.mean(s_describe), np.std(s_describe)

		print(">>>>>>>>>>>>>>>>>>>>> S RESULTS",flush=True)
		print("Mean: "+ str(s_mean),flush=True)
		#print("Average: " + str(s_arr[0]))
		print("Std: " + str(s_std))
		print(">>>>>> S Error by Point",flush=True)
		for i in range(0,total_points_used):
			print("Point "+str(i),flush=True)
			print("S: " +str(s_describe[i])+" Mean: " +str(s_mean) + " Error: " + str(s_describe[i]-s_mean),flush=True)


########################################################
if __name__ == '__main__':
	app = QApplication(sys.argv)
	window = mycalib()
	#window.show()
	sys.exit(app.exec_())