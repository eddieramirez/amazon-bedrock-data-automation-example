# Databricks notebook source
# MAGIC %md
# MAGIC # Object Detection and Visual Intelligence with Amazon Bedrock Data Automation
# MAGIC
# MAGIC Welcome to our specialized workshop on video object detection using Amazon Bedrock Data Automation (BDA). This notebook focuses on BDA's powerful object detection capabilities that transform raw video content into structured, searchable, and analyzable data.
# MAGIC
# MAGIC ## Why Object Detection Matters for Video
# MAGIC
# MAGIC Video content presents unique opportunities and challenges:
# MAGIC
# MAGIC - A single minute of video contains approximately 1,800 frames (at 30 fps)
# MAGIC - Organizations possess vast libraries of video content with limited metadata
# MAGIC - Manual object tagging is prohibitively expensive at $15-25 per minute of processed content 
# MAGIC - Traditional approaches detect objects in isolated frames without understanding temporal context
# MAGIC - Only 1-2% of video content is typically leveraged in business intelligence systems
# MAGIC
# MAGIC Amazon Bedrock Data Automation transforms this landscape by enabling intelligent object detection across video content with business-friendly field naming and visualization.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setting Up Our Environment
# MAGIC
# MAGIC Let's begin by installing required libraries and importing dependencies. We'll be using our specialized utility functions for object detection visualization.

# COMMAND ----------

# Install required packages
%pip install "boto3>=1.37.4" "matplotlib" "moviepy" "pandas" "seaborn" "wordcloud" --upgrade -qq
%restart_python 

# COMMAND ----------

# Import necessary libraries
import boto3
import json
import uuid
import time
import matplotlib.pyplot as plt
from datetime import datetime
from IPython.display import Video, clear_output, HTML, display, Markdown
import warnings
import urllib
import os
warnings.filterwarnings('ignore')

# Import utilities
from bda_object_detection_utils import BDAObjectDetectionUtils

region = "us-west-2" # Enter your region here
bucket = "bda-workshop-<YOUR_ACCOUNT_ID>-<YOUR_REGION>" # Replace placeholders with actual account ID and region

# Initialize our utility class
bda_utils = BDAObjectDetectionUtils(region, bucket)
print(f"Setup complete. BDA utilities initialized for region: {bda_utils.current_region}")
print(f"Using S3 bucket: {bda_utils.bucket_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Prepare Sample Video
# MAGIC
# MAGIC First, we'll download a sample video and upload it to S3 for processing with BDA. We'll use a short video that contains various objects that BDA can detect and analyze.

# COMMAND ----------

# Download sample video using our enhanced utility function
sample_video = 'movie-demo.mp4'
source_url = 'https://ws-assets-prod-iad-r-pdx-f3b3f9f1a7d6a3d0.s3.us-west-2.amazonaws.com/335119c4-e170-43ad-b55c-76fa6bc33719/NetflixMeridian.mp4'

# Download the video with enhanced error handling
try:
    bda_utils.download_video(source_url, sample_video)
    print(f"Successfully downloaded video to {sample_video}")
except Exception as e:
    print(f"Error downloading video: {e}")

# Display the video in the notebook for preview
display(Video(sample_video, width=800))

# Upload to S3 for BDA processing
s3_key = f'{bda_utils.data_prefix}/{sample_video}'
s3_uri = bda_utils.upload_to_s3(sample_video, s3_key)
print(f"Uploaded video to S3: {s3_uri}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Define Enhanced Blueprint for Object Detection
# MAGIC
# MAGIC Now we'll define a custom blueprint for object detection. This blueprint uses business-friendly field names that better reflect the purpose of each detection type, making the schema more intuitive and self-documenting.

# COMMAND ----------

# Define the enhanced blueprint for object detection with business-friendly field names
# and detailed comments about each field based on AWS documentation

blueprint = {
  "$schema": "http://json-schema.org/draft-07/schema#",
  "description": "This blueprint enhances the searchability and discoverability of video content by providing comprehensive object detection and scene analysis.",
  "class": "media_search_video_analysis",
  "type": "object",
  "properties": {
    # Targeted Object Detection: Identifies visually prominent objects in the video with bounding boxes
    # Set granularity to chapter level for more precise object detection
    "targeted-object-detection": {
      "items": {
        "$ref": "bedrock-data-automation#/definitions/Entity"
      },
      "type": "array",
      "instruction": "Please detect all the visually prominent objects in the video",
      "granularity": ["chapter"]  # Chapter-level granularity provides per-scene object detection
    },
    
    # Category-Based Detection: Groups objects by categories for better organization with bounding boxes
    # This allows for detecting objects belonging to specific categories (e.g., furniture)
    "category-based-detection": {
      "items": {
        "$ref": "bedrock-data-automation#/definitions/Entity"
      },
      "type": "array",
      "instruction": "Detect all the furniture items in the video",
      "granularity": ["chapter"]  # Per-scene category detection
    },
    
    # Visual Importance Analysis: Determines the most significant visual elements with bounding boxes
    # This helps identify what draws viewer attention in each scene
    "visual-importance-analysis": {
      "items": {
        "$ref": "bedrock-data-automation#/definitions/Entity"
      },
      "type": "array",
      "instruction": "Find and detect the most visually important elements in the video",
      "granularity": ["chapter"]  # Analyze visual importance per chapter
    },
    
    # Contextual Object Detection: Finds objects within specific contexts with bounding boxes
    # Allows for more complex detection scenarios like "people riding motorcycles"
    "contextual-object-detection": {
      "items": {
        "$ref": "bedrock-data-automation#/definitions/Entity"
      },
      "type": "array",
      "instruction": "Detect the people driving a car in the video",
      "granularity": ["chapter"]  # Per-chapter contextual object detection
    },
    
    # Object Verification: Confirms presence/absence of specific objects
    # Video-level granularity checks across the entire video
    "object-verification": {
      "type": "boolean",
      "inferenceType": "inferred",  # Uses inference rather than direct extraction
      "instruction": "Are there zebras in this video? Respond with false if none are present.",
      "granularity": ["video"]  # Video-level verification spans the entire content
    },
    
    # Verification Explanation: Provides reasoning for object verification
    # Helps users understand why the model determined objects were present/absent
    "verification-explanation": {
      "type": "string",
      "inferenceType": "inferred",
      "instruction": "Explain why you believe zebras are or are not present in the video",
      "granularity": ["video"]  # Video-level explanation
    },
    
    # Total Objects Count: Aggregates unique objects across the video
    # Provides a high-level metric of object diversity
    "total-objects-count": {
      "type": "number",
      "inferenceType": "inferred",
      "instruction": "Count the total number of distinct objects detected in the video",
      "granularity": ["video"]  # Video-level aggregate count
    }
  }
}

# Generate a unique blueprint name to avoid naming conflicts
unique_id = str(uuid.uuid4())[0:6]
blueprint_name = f"bda-video-enhanced-blueprint-{unique_id}"

print(f"Creating blueprint with name: {blueprint_name}")

# Create the blueprint in BDA
try:
    bp_response = bda_utils.bda_client.create_blueprint(
        blueprintName=blueprint_name,
        type='VIDEO',  # Specify this is for video analysis
        blueprintStage='LIVE',  # Use development stage for workshop
        schema=json.dumps(blueprint),  # Convert blueprint dict to JSON string
    )
    
    blueprint_arn = bp_response.get("blueprint", {}).get("blueprintArn")
    print(f"Blueprint created successfully with ARN: {blueprint_arn}")
except Exception as e:
    print(f"Error creating blueprint: {e}")
    blueprint_arn = None

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Define BDA Configuration and Create Project
# MAGIC
# MAGIC Now we'll define the standard output configuration for video analysis and create a BDA project. This configuration determines what information BDA will extract from the video.

# COMMAND ----------

# Define standard output configuration for video processing
standard_output_config = {
    'video': {
        'extraction': {
            'category': {
                'state': 'ENABLED',
                'types': [
                    'CONTENT_MODERATION',  # Detect inappropriate content
                    'TEXT_DETECTION',      # Extract text from the video
                    'TRANSCRIPT',          # Generate transcript of spoken content
                    'LOGOS'                # Identify brand logos
                ]
            },
            'boundingBox': {
                'state': 'ENABLED'         # Include bounding boxes for detected elements
            }
        },
        'generativeField': {
            'state': 'ENABLED',
            'types': [
                'VIDEO_SUMMARY',           # Generate overall video summary
                'CHAPTER_SUMMARY',         # Generate summaries for each chapter
                'IAB'                      # Classify into IAB categories
            ]
        }
    }
}

# Create a BDA project with our standard output configuration
print("Creating BDA project for object detection...")
response = bda_utils.bda_client.create_data_automation_project(
    projectName=f'bda-workshop-object-detection-project-{str(uuid.uuid4())[0:4]}',
    projectDescription='BDA workshop object detection project',
    projectStage='LIVE',
    standardOutputConfiguration=standard_output_config,
    customOutputConfiguration={
        'blueprints': [
            {
                'blueprintArn': blueprint_arn,
                'blueprintStage': 'LIVE'
            },
        ]
    }
)

# Get the project ARN
video_project_arn = response.get("projectArn")
print(f"BDA project created with ARN: {video_project_arn}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Process Video with BDA
# MAGIC
# MAGIC Now we'll use the `invoke_data_automation_async` API to process our video with BDA. BDA operates asynchronously due to the complexity and processing time required for video analysis and object detection.

# COMMAND ----------

# Invoke BDA to process the video
print(f"Processing video: {s3_uri}")
print(f"Results will be stored at: s3://{bda_utils.bucket_name}/{bda_utils.output_prefix}")

# Call the invoke_data_automation_async API
response = bda_utils.bda_runtime_client.invoke_data_automation_async(
    inputConfiguration={
        's3Uri': s3_uri  # The S3 location of our video
    },
    outputConfiguration={
        's3Uri': f's3://{bda_utils.bucket_name}/{bda_utils.output_prefix}'  # Where to store results
    },
    dataAutomationConfiguration={
        'dataAutomationProjectArn': video_project_arn,  # The project we created
        'stage': 'LIVE'                          # Must match the project stage
    },
    dataAutomationProfileArn=f'arn:aws:bedrock:{bda_utils.current_region}:{bda_utils.account_id}:data-automation-profile/us.data-automation-v1'
)

# Get the invocation ARN
invocation_arn = response.get("invocationArn")
print(f"Invocation ARN: {invocation_arn}")

# Wait for processing to complete using our enhanced pattern
status_response = bda_utils.wait_for_completion(
    get_status_function=bda_utils.bda_runtime_client.get_data_automation_status,
    status_kwargs={'invocationArn': invocation_arn},
    completion_states=['Success'],
    error_states=['ClientError', 'ServiceError'],
    status_path_in_response='status',
    max_iterations=40,  # Video might take longer than other modalities
    delay=10
)

# Check if processing was successful
if status_response['status'] == 'Success':
    output_config_uri = status_response.get("outputConfiguration", {}).get("s3Uri")
    print(f"\nVideo processing completed successfully!")
    print(f"Output configuration: {output_config_uri}")
else:
    print(f"\nVideo processing failed with status: {status_response['status']}")
    if 'error_message' in status_response:
        print(f"Error message: {status_response['error_message']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Retrieve and Explore BDA Results
# MAGIC
# MAGIC Now that the video has been processed, let's retrieve the results from S3 and explore the object detection insights.

# COMMAND ----------

# Load job metadata
config_data = bda_utils.read_json_from_s3(output_config_uri)

# Get custom output path
custom_output_path = config_data["output_metadata"][0]["segment_metadata"][0]["custom_output_path"]
result_data = bda_utils.read_json_from_s3(custom_output_path)

# Save the result data to the bda-results directory
with open('bda_results.json', 'w') as f:
    json.dump(result_data, f)
    
print(f"Saved video object detection results to: bda_results.json")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Video Metadata and Object Verification
# MAGIC
# MAGIC Let's first look at the video metadata and object verification results to understand what BDA detected across the entire video.

# COMMAND ----------

# Display video metadata and object verification results
print("=== Video Metadata and Object Verification ===\n")
metadata = result_data.get("metadata", {})
inference_result = result_data.get("inference_result", {})

print(f"Video Type: {inference_result.get('video-type', 'N/A')}")
print(f"Genre: {inference_result.get('genre', 'N/A')}")

if "object-verification" in inference_result:
    verification = inference_result["object-verification"]
    explanation = inference_result.get("verification-explanation", "No explanation provided")
    print(f"\nObject Verification Query: Are there zebras in this video?")
    print(f"Result: {'Present' if verification else 'Not Present'}")
    print(f"Explanation: {explanation}")

if "total-objects-count" in inference_result:
    print(f"\nTotal Unique Objects Detected: {int(inference_result['total-objects-count'])}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Chapter-Based Object Detection Analysis
# MAGIC
# MAGIC Let's analyze the objects detected across different chapters of the video. This provides insights into what objects appear in each scene and how they relate to the narrative.

# COMMAND ----------

# Analyze objects detected across video chapters
# Call the method from our bda_utils instance instead of as a standalone function
bda_utils.analyze_chapter_objects(result_data)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Object Detection with Bounding Boxes
# MAGIC
# MAGIC Now we'll visualize the objects detected in a specific chapter with their bounding boxes. This provides precise spatial information about where objects appear in the video frames.

# COMMAND ----------

# Choose a chapter to analyze (index starts at 0)
chapter_index = 2

# Visualize objects with bounding boxes for the selected chapter
bda_utils.visualize_objects_with_bounding_boxes(sample_video, result_data, chapter_index, confidence_threshold=0.6)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Conclusion: The Business Value of Object Detection
# MAGIC
# MAGIC Amazon Bedrock Data Automation's object detection capabilities transform raw video assets into structured, searchable data that delivers significant business value across industries:
# MAGIC
# MAGIC ### Media & Entertainment
# MAGIC - **Content Discovery**: Enable search and discovery by specific objects, improving viewer engagement
# MAGIC - **Automated Metadata**: Generate rich object-based metadata without manual tagging
# MAGIC - **Content Recommendations**: Match content based on similar visual elements and objects
# MAGIC - **Smart Navigation**: Allow viewers to jump directly to scenes containing objects of interest
# MAGIC
# MAGIC ### Marketing & Advertising
# MAGIC - **Product Placement**: Track product appearances and measure screen time
# MAGIC - **Competitive Analysis**: Analyze competitors' video content for featured products and objects
# MAGIC - **Brand Monitoring**: Automatically detect logo and product appearances
# MAGIC - **Contextual Targeting**: Place ads alongside content featuring similar objects
# MAGIC
# MAGIC ### Retail & E-Commerce
# MAGIC - **Shoppable Content**: Tag products in videos for direct purchasing
# MAGIC - **Visual Merchandising**: Analyze store layout and product placement
# MAGIC - **Product Detection**: Identify products in user-generated content
# MAGIC - **Visual Search**: Enable search for products seen in videos
# MAGIC
# MAGIC ### Security & Compliance
# MAGIC - **Object Verification**: Confirm presence or absence of required safety equipment
# MAGIC - **Suspicious Object Detection**: Flag potentially dangerous items
# MAGIC - **Prohibited Content**: Identify banned or restricted objects
# MAGIC - **Safety Monitoring**: Ensure compliance with safety regulations
# MAGIC
# MAGIC By leveraging object detection with business-friendly visualizations, organizations can unlock the full value of their video content libraries and create more engaging, discoverable, and monetizable video experiences.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Clean up
# MAGIC
# MAGIC Delete the BDA project, blueprint, image, and result from S3.

# COMMAND ----------

# delete BDA project
response = bda_utils.bda_client.delete_data_automation_project(
    projectArn=video_project_arn
)
response

# COMMAND ----------

# delete BDA blueprint
response = bda_utils.bda_client.delete_blueprint(
    blueprintArn=blueprint_arn
)
response

# COMMAND ----------

# delete uploaded image from S3
bda_utils.s3_client.delete_object(Bucket=bucket, Key=s3_key)

# COMMAND ----------

# delete results from S3
bda_utils.delete_s3_folder(urllib.parse.urlparse(output_config_uri.replace("job_metadata.json","")).path.lstrip("/"))