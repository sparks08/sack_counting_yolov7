import time
import os
import boto3
import datetime
from dateutil import parser
from natsort import natsorted
from moviepy.editor import VideoFileClip, concatenate_videoclips

now = (datetime.datetime.today() - datetime.timedelta(hours=1, minutes=0))
current_time = datetime.datetime.strftime(now, '%H:%M')
date_current = datetime.datetime.strftime(now, '%d/%m/%y')
date_time = parser.parse(current_time)

client = boto3.client('s3', aws_access_key_id='AKIATVLJHPSRQZPX2QLY',
                      aws_secret_access_key='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', region_name="af-south-1")

paginator = client.get_paginator('list_objects')


global_list = list()


def get_video():

    cameraID = input("Enter Camera ID")
    directory = '/home/inqholduser1/Gaurav/stream_video/'
    for f in os.listdir(directory):
        os.remove(os.path.join(directory, f))
    
    t = time.localtime()
    current_time = time.strftime("%H:%M:%S", t)
    print(current_time,"Video extraction started")
    result = paginator.paginate(Bucket='inq-shayona', Prefix=f'{cameraID}')     #1666071948649985
    files = []
    
    for page in result:
        if "Contents" in page:
            for i in page["Contents"]:
                path_list = list()
                date_server_creation = datetime.datetime.strftime(i['LastModified'], '%d/%m/%y')
                server_time_creation = datetime.datetime.strftime(i['LastModified'], '%H:%M')
                server_time = parser.parse(server_time_creation)
                if date_current == date_server_creation:
                    if server_time > date_time:
                        print(f'Server Time is {server_time} and current date is {date_time}')
                        obj, filename = i['Key'].split("/")
                        files.append(filename)
    print("Files fetched")
                        
    for file in files:
    	client.download_file('inq-shayona', f'{cameraID}/{file}',
                                         directory + file)
    print("All files download")

    currentVideo = None
    for filePath in natsorted(os.listdir(path)):
        try:
            if filePath.endswith(".mp4"):
                if currentVideo == None:
                    currentVideo = VideoFileClip(path + filePath)
                    continue
                video_2 = VideoFileClip(path + filePath)
                currentVideo = concatenate_videoclips([currentVideo, video_2])
        except Exception as e:
            print(e)

    currentVideo.write_videofile("/home/inqholduser1/Gaurav/stream_video/stream.mp4")
    print("Videos concatenated")

