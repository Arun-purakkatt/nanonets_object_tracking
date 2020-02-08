import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
import random
from PIL import Image
import torch
from torch.autograd import Variable
import PIL.ImageOps    
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import cv2,pickle,sys
import argparse

from deepsort import *


def get_gt(image,frame_id,gt_dict):

	if frame_id not in gt_dict.keys() or gt_dict[frame_id]==[]:
		return None,None,None

	frame_info = gt_dict[frame_id]

	detections = []
	ids = []
	out_scores = []
	labels = []
	for i in range(len(frame_info)):

		coords = frame_info[i]['coords']

		x1,y1,w,h = coords
		x2 = x1 + w
		y2 = y1 + h

		xmin = min(x1,x2)
		xmax = max(x1,x2)
		ymin = min(y1,y2)
		ymax = max(y1,y2)	

		detections.append([x1,y1,w,h])
		out_scores.append(frame_info[i]['conf'])
		labels.append(frame_info[i]['label'])

	return detections,out_scores,labels


def get_dict(filename):
	with open(filename) as f:	
		d = f.readlines()

	d = list(map(lambda x:x.strip(),d))

	last_frame = int(d[-1].split(',')[0])

	gt_dict = {x:[] for x in range(last_frame+1)}

	for i in range(len(d)):
		a = list(d[i].split(','))
		label = a[-1]
		a = list(map(float,a[:-1]))

		if a[6] < 0.8: continue

		coords = a[2:6]
		confidence = a[6]
		gt_dict[a[0]].append({'coords':coords,'conf':confidence, 'label':label})

	return gt_dict

def withinbbox(pt, bbox):
	x = (pt[0] + pt[2]) / 2
	y = (pt[1] + pt[3]) / 2
	if bbox[0] <= x <= bbox[0] + bbox[2] and bbox[1] <= y <= bbox[1] + bbox[3]:
		return True
	else:
		return False

parser = argparse.ArgumentParser()
parser.add_argument('--det_file', default='det/video1_0_dets.txt', type=str, help='path to dets')
parser.add_argument('--in_vid_file', default='video1_0.avi', type=str, help='path to input video file')
parser.add_argument('--out_vid_file', default='video1_0_tracked.avi', type=str, help='path to output video file')

if __name__ == '__main__':
	args = parser.parse_args()

	#Load detections for the video. Options available: yolo,ssd and mask-rcnn
	filename = args.det_file
	gt_dict = get_dict(filename)

	cap = cv2.VideoCapture(args.in_vid_file)
	width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	frames_per_second = cap.get(cv2.CAP_PROP_FPS)

	#Initialize deep sort.
	deepsort = deepsort_rbc()

	frame_id = 1

	out = cv2.VideoWriter(
		filename=args.out_vid_file,
		fourcc=cv2.VideoWriter_fourcc(*'DIVX'),
		fps=float(frames_per_second),
		frameSize=(width + width//3, height),
		isColor=True,
	)

	r, f = cap.read()
	rects = []
	answer = 'y'
	while answer == 'y':
		bbox = cv2.selectROI('DRAW BOUNDING BOXES HERE', f, showCrosshair=False)
		rects.append(bbox)
		answer = input("DO YOU WISH TO DRAW ANOTHER BOUNDING BOX? (y=yes)")

	labelDict = {}
	while True:
		print(frame_id)		

		ret,frame = cap.read()
		if ret is False:
			frame_id+=1
			break	

		frame = frame.astype(np.uint8)

		detections,out_scores,labels = get_gt(frame,frame_id,gt_dict)

		if detections is None:
			print("No dets")
			frame_id+=1
			continue

		detections = np.array(detections)
		out_scores = np.array(out_scores) 

		tracker,detections_class = deepsort.run_deep_sort(frame,out_scores,detections,labels)

		counterImg = np.zeros((height, width//3, 3), np.uint8)
		for track in tracker.tracks:
			if not track.is_confirmed() or track.time_since_update > 1:
				continue
			
			bbox = track.to_tlbr() #Get the corrected/predicted bounding box
			id_num = str(track.track_id) #Get the ID for the particular track.
			features = track.features #Get the feature vector corresponding to the detection.

			#Draw bbox from tracker.
			cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2)
			cv2.putText(frame, str(id_num),(int(bbox[0]), int(bbox[1])),0, 5e-3 * 200, (0,255,0),2)
			cv2.putText(frame, track.label, (int(bbox[0]), int(bbox[1])-30), 0, 5e-3 * 200, (0, 255, 0), 2)

			# draw boxes for counting
			for rect in rects:
				x = int(rect[0])
				y = int(rect[1])
				w = int(rect[2])
				h = int(rect[3])
				cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

				if withinbbox(bbox, rect):
					if track.label in labelDict.keys():
						if id_num not in labelDict[track.label]:
							labelDict[track.label].append(id_num)
					else:
						labelDict[track.label] = []
						labelDict[track.label].append(id_num)

				verticalCounter = 50
				for label, IDlist in labelDict.items():
					cv2.putText(counterImg, label + ': ' + str(len(IDlist)),
								(100, verticalCounter), 0, 1, (0, 255, 0), 2)
					verticalCounter += 50

			#Draw bbox from detector. Just to compare.
			for det in detections_class:
				bbox = det.to_tlbr()
				cv2.rectangle(frame,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,0), 2)

		finalImg = np.concatenate((frame, counterImg), axis=1)
		
		cv2.imshow('frame',finalImg)
		out.write(finalImg)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

		frame_id+=1
	out.release()
