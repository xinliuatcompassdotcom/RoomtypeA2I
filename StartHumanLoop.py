#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 3 09:00:34 2020

@author: xin.liu
"""

REGION = 'us-east-1'
CUSTOM_HUMAN_LOOP_NAME = 'a2i-custom-1' # This is idempotency token. Update this everytime when running this notebook

import boto3
#import io
import json
#import uuid
#import botocore
import time
from datetime import datetime

a2i = boto3.client('sagemaker-a2i-runtime', REGION)
sagemaker = boto3.client('sagemaker', REGION)
rekognition= boto3.client('rekognition', REGION)
textract = boto3.client('textract', REGION)

def findRoomType(img_url='https://compass-development-ai-xin.s3.amazonaws.com/RoomTypeExampleImages/livingroom.jpg',
             endpoint_name='lens-encoding-production'):
    '''
    classify an image by calling a SageMaker Endpoint.    

    Parameters
    ----------
    img_url : TYPE string. image URL to be classified. The default value provides an example.
    endpoint_name : TYPE string. The SageMaker endpoint name. The default value provides an example.

    Returns
    -------
    valid : TYPE Boolean. Whether the returned result is valid or not.
    roomType : TYPE string. One of {'bathroom', 'bedroom', 'dining_room', 'Exterior', 'Interior', 'kitchen', 'living_room'}
    roomTypeScore : TYPE float. Confidence of the classification. 0.0 means not confident at all. 1.0 means extremely confident.
    '''
    print('Request for: {}'.format(img_url))
    
    runtime = boto3.Session().client(service_name='sagemaker-runtime', region_name='us-east-1')
    request = {'url': img_url}
    start_time = time.time()
    response = runtime.invoke_endpoint(EndpointName=endpoint_name, ContentType='application/json', Body=json.dumps(request))
    elapsed_time = time.time() - start_time
    print('Response received in {} s'.format(elapsed_time))
    body = response['Body'].read()
    dictBody = json.loads(body)
    valid = dictBody['valid']
    roomType = dictBody['roomType']
    roomTypeScore = dictBody['roomTypeScore']
    return (valid, roomType, roomTypeScore)

def startHumanLoop(flowArn='arn:aws:sagemaker:us-east-1:149465543054:flow-definition/xin-roomtype-8classes-a2i',
                   inputURL='https://compass-development-ai-xin.s3.amazonaws.com/RoomTypeExampleImages/livingroom.jpg'):
    '''
    start human loop

    Parameters
    ----------
    flowArn : TYPE string. The human workflow ARN. The default value provides an example.
    inputURL : TYPE string. A URL pointing to the image that was classified. The default value provides an example.

    Returns
    -------
    None.

    '''    
    ## create a unique human loop name
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y-%H-%M-%S-%f")
    humanLoopName = dt_string
    
    a2i = boto3.client('sagemaker-a2i-runtime', REGION)
    
    a2i.start_human_loop(
            HumanLoopName=humanLoopName,
            FlowDefinitionArn=flowArn, 
            HumanLoopInput= {
                'InputContent':'{\"taskObject\":\"' + inputURL + '", \"foo\":\"bar" }' 
            }
    )
    # Do work on private workteam

def sendNotification(snsTopicArn='arn:aws:sns:us-east-1:149465543054:AmazonSageMaker-xin-a2i-sns-topic',
                     labelingPageURL='https://pwqqomw3ix.labeling.us-east-1.sagemaker.aws/'):
    '''
    Send notification to labeling team, e.g., when new task is available.
    Please be advised that A2I won't notify the labeling team automatically when a new human task is started.

    Parameters
    ----------
    snsTopicArn : TYPE string. AWS SNS Topic ARN. The default value provides an example.
    labelingPageURL : TYPE string. URL pointing to the labeling page. The default value provides an example.

    Returns
    -------
    None.
    '''
  
    emailText = 'From A2I: New labeling task is abailable at {}'.format(labelingPageURL)
    
    client3 = boto3.client('sns', region_name='us-east-1')
    response = client3.publish(TopicArn=snsTopicArn, Message = emailText)
    print('notification sent to label team')

def roomtypeClassificationWithHumanLoop(imgUrl='https://compass-development-ai-xin.s3.amazonaws.com/RoomTypeExampleImages/livingroom.jpg',
                                        endpointName='lens-encoding-production',
                                        threshold = 0.5,
                                        flowArn='arn:aws:sagemaker:us-east-1:149465543054:flow-definition/xin-roomtype-8classes-a2i',
                                        snsTopicArn='arn:aws:sns:us-east-1:149465543054:AmazonSageMaker-xin-a2i-sns-topic',
                                        labelingPageURL='https://pwqqomw3ix.labeling.us-east-1.sagemaker.aws/'):
    '''
    This function provides an example of the entire workflow, from image classification
      to starting Human Loop when the confidence is lower than a threhold,
      and finally sending a notification to the labeling team.
 
    Parameters
    ----------
    img_url : TYPE string. image URL to be classified. The default value provides an example.
    endpoint_name : TYPE string. The SageMaker endpoint name. The default value provides an example.
    threshold : TYPE float. the threshold determining when to start human loop.
    flowArn : TYPE string. AWS SNS Topic ARN. The default value provides an example.
    snsTopicArn : TYPE string. AWS SNS Topic ARN. The default value provides an example.
    labelingPageURL : TYPE string. URL pointing to the labeling page. The default value provides an example.

    Returns
    -------
    None.
    '''

    valid, roomType, roomTypeScore = findRoomType(img_url=imgUrl,
                                              endpoint_name=endpointName)
    if (valid and roomTypeScore < threshold):
        print('roomtype: {}, score: {}'.format(roomType, roomTypeScore))
        startHumanLoop(flowArn, imgUrl)
        sendNotification(snsTopicArn, labelingPageURL)
    else:
        print('invalid roomtype classification')
    
if __name__ == '__main__':
    ## defining all parameters
    imgUrl='https://compass-development-ai-xin.s3.amazonaws.com/RoomTypeExampleImages/livingroom.jpg'
    endpointName='lens-encoding-production'
    flowArn='arn:aws:sagemaker:us-east-1:149465543054:flow-definition/xin-roomtype-8classes-a2i'
    snsTopicArn='arn:aws:sns:us-east-1:149465543054:AmazonSageMaker-xin-a2i-sns-topic'
    labelingPageURL='https://pwqqomw3ix.labeling.us-east-1.sagemaker.aws/'

    ## room type classification with Human Loop
    roomtypeClassificationWithHumanLoop(imgUrl, endpointName, 1.0, flowArn, snsTopicArn, labelingPageURL)