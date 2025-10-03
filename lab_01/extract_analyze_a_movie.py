# Databricks notebook source
# MAGIC %md
# MAGIC # Extract and analyze a movie

# COMMAND ----------

# MAGIC %md
# MAGIC Industries such as Media & Entertainment, Advertising and Sports manage vast inventories of professionally produced videos, including TV shows, movies, news, sports events, documentaries, and more. To effectively extract insights from this type of video content, users require information such as video summaries, chapter-level analysis, IAB classifications for ad targeting, and speaker identification.
# MAGIC
# MAGIC > [IAB categories](https://smartclip.tv/adtech-glossary/iab-categories/) are standard classifications for web content that are developed by the Interactive Advertising Bureau (IAB). These categories are used to sort advertisers into industries and segments.

# COMMAND ----------

# MAGIC %md
# MAGIC In this lab, we will use BDA Video to extract and analyze a sample open-source movie [Meridian](https://en.wikipedia.org/wiki/Meridian_(film)), walking through the process and exploring the generated outputs.
# MAGIC ![video moderation](../static/bda-video-chapter.png)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prerequisites

# COMMAND ----------

# MAGIC %pip install "boto3>=1.37.4" "moviepy==2.1.2" --upgrade -qq
# MAGIC %restart_python 

# COMMAND ----------

import boto3
import json
import uuid
import utils
import urllib.parse

region = "us-west-2" # Enter your region here

session = boto3.session.Session(region_name=region)
bda_client = session.client('bedrock-data-automation')
bda_runtime_client = session.client('bedrock-data-automation-runtime')
s3_client = session.client('s3')

# COMMAND ----------

# MAGIC %md
# MAGIC We recommend creating a new S3 bucket in the same region where you plan to run the workshop. You can name it `bda-workshop-YOUR_ACCOUNT_ID-YOUR_REGION`.

# COMMAND ----------

data_bucket = "bda-workshop-<YOUR_ACCOUNT_ID>-<YOUR_REGION>" # Replace placeholders with actual account ID and region
data_prefix = "bda-workshop/video"
output_prefix = "bda-workshop/video/output"


try:
    s3_client.head_bucket(Bucket=data_bucket)
    print(f"Workshop S3 bucket: {data_bucket}")
except:
    print(f"Error: Bucket Does not Exist: {data_bucket}")
    raise

# COMMAND ----------

# Get current AWS account Id and region
sts = session.client('sts')
account_id = sts.get_caller_identity()['Account']

print(f'Current AWS account Id: {account_id}, region name: {region}')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create a BDA project with a standard output configuration for videos
# MAGIC To start a BDA job, you need a BDA project, which organizes both standard and custom output configurations. This project is reusable, allowing you to apply the same configuration to process multiple videos that share the same settings.

# COMMAND ----------

# MAGIC %md
# MAGIC In the code snippet below, we create a BDA project with standard output configurations for video modality. These configurations can be tailored to extract only the specific information you need. In this lab, we will enable the below video outputs:
# MAGIC - Full video summary
# MAGIC - Chapter summaries
# MAGIC - IAB categories on the chapter level
# MAGIC - Full audio transcript
# MAGIC - Text in video with bounding-boxes
# MAGIC - Brand & logo in video with bounding-boxes
# MAGIC
# MAGIC For a complete API reference for creating a BDA project, refer to this [document](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock-data-automation/client/create_data_automation_project.html).

# COMMAND ----------

response = bda_client.create_data_automation_project(
    projectName=f'bda-workshop-video-project-{str(uuid.uuid4())[0:4]}',
    projectDescription='BDA workshop video sample project',
    projectStage='DEVELOPMENT',
    standardOutputConfiguration={
        'video': {
            'extraction': {
                'category': {
                    'state': 'ENABLED',
                    'types': ['TEXT_DETECTION','TRANSCRIPT','LOGOS'],
                },
                'boundingBox': {
                    'state': 'ENABLED',
                }
            },
            'generativeField': {
                'state': 'ENABLED',
                'types': ['VIDEO_SUMMARY','CHAPTER_SUMMARY','IAB'],
            }
        }
    }
)

# COMMAND ----------

# MAGIC %md
# MAGIC The create_data_automation_project API will return the project ARN, which we will use it to invoke the video analysis task.

# COMMAND ----------

video_project_arn = response.get("projectArn")
print("BDA video project ARN:", video_project_arn)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Start an asynchronous BDA task to extract and analyze a movie
# MAGIC In this section, we will use a open-source movie Meridian, and extract and analyze it using BDA, applying the configuration defined in the BDA project. We will then review the output to gain a deeper understanding of how BDA performs video extraction and analysis.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Prepare the sample video

# COMMAND ----------

# Download sample video
sample_video_movie = 'NetflixMeridian.mp4'
source_url = f'https://ws-assets-prod-iad-r-pdx-f3b3f9f1a7d6a3d0.s3.us-west-2.amazonaws.com/335119c4-e170-43ad-b55c-76fa6bc33719/NetflixMeridian.mp4'

!curl {source_url} --output {sample_video_movie}

# COMMAND ----------

# MAGIC %md
# MAGIC Let's display the video. [Meridian](https://en.wikipedia.org/wiki/Meridian_(film)) is a test movie from Netflix, we use it to showcase how BDA works with video extraction. As you can see, it is a classic-style movie composed of multiple chapters.

# COMMAND ----------

from IPython.display import Video
Video(sample_video_movie, width=800)

# COMMAND ----------

# MAGIC %md
# MAGIC To analyze the video using BDA, we need to upload it to an S3 bucket that BDA can access. 

# COMMAND ----------

s3_key = f'{data_prefix}/{sample_video_movie.split("/")[-1]}'
s3_client.upload_file(sample_video_movie, data_bucket, s3_key)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Start BDA task
# MAGIC We will now invoke the BDA API to process the uploaded video. You need to provide the BDA project ARN that we created at the beginning of the lab and specify an S3 location where BDA will store the output results.
# MAGIC
# MAGIC For a complete API reference for invoke a BDA async task, refer to this [document](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock-data-automation-runtime/client/invoke_data_automation_async.html).

# COMMAND ----------

response = bda_runtime_client.invoke_data_automation_async(
    inputConfiguration={
        's3Uri': f's3://{data_bucket}/{s3_key}'
    },
    outputConfiguration={
        's3Uri': f's3://{data_bucket}/{output_prefix}'
    },
    dataAutomationConfiguration={
        'dataAutomationProjectArn': video_project_arn,
        'stage': 'DEVELOPMENT'
    },
    notificationConfiguration={
        'eventBridgeConfiguration': {
            'eventBridgeEnabled': False
        }
    },
    dataAutomationProfileArn=f'arn:aws:bedrock:{region}:{account_id}:data-automation-profile/us.data-automation-v1'
)

# COMMAND ----------

# MAGIC %md
# MAGIC The `invoke_data_automation_async` API is asynchronous. It returns an invocation task identifier, `invocationArn`. We can then use another API `get_data_automation_status` to monitor the task's status until it completes.
# MAGIC
# MAGIC > In production workloads, an event-driven pattern is recommended. Allow BDA to trigger the next step once the task is complete. This can be achieved by configuring the notificationConfiguration in the invoke task, which will send a notification to a subscribed AWS service, such as a Lambda function. Alternatively, you can set up an S3 trigger on the bucket where BDA will drop the results.

# COMMAND ----------

invocation_arn = response.get("invocationArn")
print("BDA task started:", invocation_arn)

# COMMAND ----------

# MAGIC %md
# MAGIC In this lab, we will use the loop below to monitor the task by calling the `get_data_automation_status` API every 5 seconds until the task is complete.
# MAGIC
# MAGIC This video will take ~5-10 minutes.

# COMMAND ----------

import time
from IPython.display import clear_output
from datetime import datetime

status, status_response = None, None
while status not in ["Success","ServiceError","ClientError"]:
    status_response = bda_runtime_client.get_data_automation_status(
        invocationArn=invocation_arn
    )
    status = status_response.get("status")
    clear_output(wait=True)
    print(f"{datetime.now().strftime('%H:%M:%S')} : BDA video task: {status}")
    time.sleep(5)

output_config = status_response.get("outputConfiguration",{}).get("s3Uri")
print("Ouput configureation file:", output_config)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Access the BDA analysis result
# MAGIC The `get_data_automation_status` API returns an S3 URI containing the result configuration, which provides the S3 location where BDA outputs the extraction results. We will then parse this file to retrieve the result path.

# COMMAND ----------

config_data = utils.read_json_on_s3(output_config,s3_client)
print(json.dumps(config_data, indent=4))

# COMMAND ----------

# MAGIC %md
# MAGIC As shown above, the BDA output configuration file contains metadata about the BDA result, including the job ID, status, modality, and the S3 location of the actual result JSON. We will now download this result file to verify the output.

# COMMAND ----------


result_uri = config_data["output_metadata"][0]["segment_metadata"][0]["standard_output_path"]
result_data = utils.read_json_on_s3(result_uri,s3_client)

print(json.dumps(result_data, indent=4))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Review the result
# MAGIC The BDA video analysis result contains a detailed breakdown of information, organized by video and chapter level.
# MAGIC > A video chapter is a sequence of shots that form a coherent unit of action or narrative within the video. This feature breaks down the video into meaningful segments based on visual and audible cues, provides timestamps for those segments, and summarizes each. 

# COMMAND ----------

# MAGIC %md
# MAGIC ### Full video summary
# MAGIC
# MAGIC Let's take a look at the video level summary - it distills the key themes, events, and information presented throughout the video into a concise summary. 

# COMMAND ----------

print(result_data["video"]["summary"])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Full video transcription
# MAGIC At the video level, we also receive the full transcript based on the video's audio, with speakers identified.

# COMMAND ----------

print(result_data["video"]["transcript"]["representation"]["text"])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Chapter time ranges, summaries, and IAB categories 
# MAGIC BDA also generates a chapter-level summary, as specified in the project configuration. Additionally, we get more metadata, including the start and end times of each chapter, as well as the [IAB](https://en.wikipedia.org/wiki/Interactive_Advertising_Bureau) categories classified based on the chapter content.

# COMMAND ----------

for chapter in result_data["chapters"]:
    iabs = []
    if chapter.get("iab_categories"):
        for iab in chapter["iab_categories"]:
            iabs.append(iab["category"])
        
    print(f'[{chapter["start_timecode_smpte"]} - {chapter["end_timecode_smpte"]}] {", ".join(iabs)}')
    print(chapter["summary"])
    print()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Audio transcript segements
# MAGIC Granular transcripts are also available at the chatper level. Under each chatper, you can find a list named `audio_segments` with associated timestamps. This can support additional downstream analysis that requires detailed transcript information.

# COMMAND ----------

for chapter in result_data["chapters"]:
    for trans in chapter["audio_segments"]:
        print(f'[{trans["start_timestamp_millis"]/1000} - {trans["end_timestamp_millis"]/1000}] {trans["text"]}')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Frame level text extraction with bounding-boxes and confidence scores
# MAGIC Text extraction, along with bounding boxes and confidence scores, is available at the frame level. In the output JSON structure, frames are organized under each chatper with captured timestamp. If text is detected at a given frame, you can find text_words and text_lines included at the frame level.
# MAGIC
# MAGIC Let's plot the frames for a given chapter with detected text, including their bounding boxes.

# COMMAND ----------

# plot all frames with boundingbox in the given chapter
chapter_index = 1
utils.plot_text(sample_video_movie, result_data, chapter_index)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Frame level logo detection with bounding-boxes and confidence scores
# MAGIC Logo detection is available at the frame level, in a format similar to the text detection output. The metadata is provided at the timestamp level, including bounding boxes and confidence scores.
# MAGIC
# MAGIC Let's plot the frames for a given chapter with detected logos, including their bounding boxes.

# COMMAND ----------

# plot all frames with boundingbox in the given chapter
chapter_index = 14
utils.plot_logo(sample_video_movie, result_data, chapter_index)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Shots in video
# MAGIC The BDA video returns a list of shots, each with a start and end time.
# MAGIC > A shot is a series of interrelated consecutive pictures taken contiguously by a single camera and representing a continuous action in time and space. 
# MAGIC
# MAGIC In the following cells, we will display screenshots of all the shots, based on the start timestamps from the BDA output.

# COMMAND ----------

# Generate shot images
images = utils.generate_shot_images(sample_video_movie, result_data, image_width=120)

# Ploat the shot images
utils.plot_shots(images)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC BDA video returns highly detailed metadata managed by the configuration. In this lab, we have enabled standard outputs required for media video analysis, using a movie as an example. You can explore the output JSON to discover more details. This lab does not cover moderation detection and analysis; for that, you can refer to the next lab, which uses a social media-style video as an example to better understand the moderation analysis offered by BDA video.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Clean up
# MAGIC
# MAGIC Delete the BDA project, blueprint, image, and result from S3.

# COMMAND ----------

# delete BDA project
response = bda_client.delete_data_automation_project(
    projectArn=video_project_arn
)
response

# COMMAND ----------

# delete uploaded image from S3
s3_client.delete_object(Bucket=data_bucket, Key=s3_key)

# COMMAND ----------

# delete results from S3
utils.delete_s3_folder(data_bucket, urllib.parse.urlparse(output_config.replace("job_metadata.json","")).path.lstrip("/") ,s3_client)