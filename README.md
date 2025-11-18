# Social Distancing Detection using Pose Estimation (Jetson Nano)
This project implements a real-time social distancing detection system using the NVIDIA Jetson Nano and the jetson-inference library. It uses pose estimation (poseNet) to detect people in a video stream, compute distances between individuals, identify violations, and generate structured logs for further analysis.

# Features
Real-time pose estimation powered by poseNet
Center-point extraction for each detected person
Pixel-distance computation between every pair
Social-distancing violation detection
Clear visual overlays (red = violation, green = safe)
Optional video recording
CSV logging for analysis
Fully optimized for Jetson Nano hardware

# Requirements
Jetson Nano (2GB or 4GB)
JetPack installed
jetson-inference library
Python 3
OpenCV
USB or CSI camera

# Project Structure
Social-Distancing-Detection/
 ├── social_distancce.py        # Main script
 ├── violations_log.csv         # Sample log file
 └── distancing_output.mp4      # Optional output video

# How to Run
 1. Webcam input
python3 social_distancce.py \
    --source=/dev/video0 \
    --output=display://0 \
    --dist-thresh=120 \
    --csv=violations_log.csv \
    --max-fps=10
2. Video file input
python3 social_distancce.py \
    --source=hallway.mp4 \
    --output=distancing_output.mp4 \
    --csv=violations_log.csv
3. Optional: real-world distance estimation
If your camera’s pixels-per-meter (ppm) is available:
--ppm=300

# Distance Calculation Workflow
Detect persons using poseNet
Extract keypoints (hips, shoulders)
Compute a central reference point for each individual
Calculate Euclidean distance:
distance = sqrt((x2 - x1)^2 + (y2 - y1)^2)
Compare with threshold to determine violation
Visualize results:
Red line → violation
Green line → safe
White circle → person center

# CSV Log Format
Column	           Description
timestamp	         Detection time
num_people	       Total detected people
pair_i	           Index of person 1
pair_j	           Index of person 2
pixel_distance	   Distance in pixels
approx_meters	     Estimated meters (if ppm is supplied)
violation	         1 = violation, 0 = safe

# Example Output
People detected: 2
Distance between 0 & 1 = 92px -> VIOLATION

# Applications
Classroom and laboratory monitoring
Workplace and warehouse safety
Public area supervision
Research and academic computer vision projects
Queue and crowd measurement systems

# Author
Sruthi Velpula
Graduate Data Science Student – University of Maryland, Baltimore County
Applied AI and Jetson Nano Projects
Email: sruthivelpula4@gmail.com
GitHub: https://github.com/SruthiVelpula
