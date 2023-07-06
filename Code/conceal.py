import ast

import cv2
import numpy as np
import pandas as pd



results = pd.read_csv('C:/Users/Akshayy/Desktop/Nerd/Dev/YOLO/License_Plate/Code/test_interpolated_train2.csv')

# load video
video_path = 'C:/Users/Akshayy/Desktop/Nerd/Dev/YOLO/License_Plate/sample.mp4'
cap = cv2.VideoCapture(video_path)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Specify the codec
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('./out_concealed_train2.mp4', fourcc, fps, (width, height))

# license_plate = {}
# for car_id in np.unique(results['car_id']):
#     max_ = np.amax(results[results['car_id'] == car_id]['license_number_score'])
#     license_plate[car_id] = {'license_crop': None,
#                              'license_plate_number': results[(results['car_id'] == car_id) &
#                                                              (results['license_number_score'] == max_)]['license_number'].iloc[0]}
#     cap.set(cv2.CAP_PROP_POS_FRAMES, results[(results['car_id'] == car_id) &
#                                              (results['license_number_score'] == max_)]['frame_nmr'].iloc[0])
#     ret, frame = cap.read()

#     x1, y1, x2, y2 = ast.literal_eval(results[(results['car_id'] == car_id) &
#                                               (results['license_number_score'] == max_)]['license_plate_bbox'].iloc[0].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))

#     license_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
#     license_crop = cv2.resize(license_crop, (int((x2 - x1) * 400 / (y2 - y1)), 400))

#     license_plate[car_id]['license_crop'] = license_crop


frame_nmr = -1

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# read frames
ret = True
while ret:
    ret, frame = cap.read()
    frame_nmr += 1
    if ret:
        df_ = results[results['frame_nmr'] == frame_nmr]
        for row_indx in range(len(df_)):
            
            # draw license plate
            x1, y1, x2, y2 = ast.literal_eval(df_.iloc[row_indx]['license_plate_bbox'].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
            
            frame = cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),0)
            frame[int(y1):int(y2),int(x1):int(x2)] = cv2.medianBlur(frame[int(y1):int(y2),int(x1):int(x2)],35)
            
        out.write(frame)
        frame = cv2.resize(frame, (1280, 720))

        # cv2.imshow('frame', frame)
        # cv2.waitKey(0)

out.release()
cap.release()