#!/usr/bin/env python3
# Social Distancing with poseNet (jetson-inference)

import sys, os, math, time, csv, argparse
import jetson.inference
import jetson.utils

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--model", default="resnet18-body", help="pose model (e.g. resnet18-body, densenet121-body)")
parser.add_argument("--threshold", type=float, default=0.15, help="pose keypoint confidence threshold")
parser.add_argument("--source", default="/dev/video0", help="camera path or video file/URL")
parser.add_argument("--output", default="display://0", help="display://0 or file sink (e.g., file://out.mp4)")
parser.add_argument("--dist-thresh", type=float, default=120.0, help="pixel distance threshold for violation")
parser.add_argument("--ppm", type=float, default=0.0, help="pixels-per-meter (if >0, prints meters too)")
parser.add_argument("--csv", default="", help="optional CSV log file path")
parser.add_argument("--no-skeleton", action="store_true", help="don’t draw pose skeleton")
parser.add_argument("--max-fps", type=float, default=0.0, help="limit processing FPS (0 = unlimited)")
args, _ = parser.parse_known_args()

def build_keypoint_index_map(net):
    m = {}
    try:
        n = net.GetNumKeypoints()
        for i in range(n):
            m[net.GetKeypointName(i).lower()] = i
    except Exception:
        pass
    return m

def try_get(kp):
    try:
        return (float(kp.x), float(kp.y), float(kp.Confidence))
    except Exception:
        return None

def hip_center_for_pose(pose, name_to_id):
    kps = pose.Keypoints
    left_id = right_id = None
    if name_to_id:
        for name, idx in name_to_id.items():
            if "left_" in name and "hip" in name: left_id = idx
            if "right_" in name and "hip" in name: right_id = idx

    def first_match(substrs):
        for kp in kps:
            try:
                nm = kp.Name.lower()
                if all(s in nm for s in substrs): return kp
            except Exception:
                pass
        return None

    L = kps[left_id] if (left_id is not None and left_id < len(kps)) else first_match(["left","hip"])
    R = kps[right_id] if (right_id is not None and right_id < len(kps)) else first_match(["right","hip"])
    L = try_get(L) if L is not None else None
    R = try_get(R) if R is not None else None
    if L and R and L[2] >= args.threshold and R[2] >= args.threshold:
        return ((L[0]+R[0])/2.0, (L[1]+R[1])/2.0)

    Ls = first_match(["left","shoulder"])
    Rs = first_match(["right","shoulder"])
    Ls = try_get(Ls) if Ls is not None else None
    Rs = try_get(Rs) if Rs is not None else None
    if Ls and Rs and Ls[2] >= args.threshold and Rs[2] >= args.threshold:
        return ((Ls[0]+Rs[0])/2.0, (Ls[1]+Rs[1])/2.0)

    xs=ys=cnt=0
    for kp in kps:
        v = try_get(kp)
        if v and v[2] >= args.threshold:
            xs += v[0]; ys += v[1]; cnt += 1
    if cnt>0: return (xs/cnt, ys/cnt)
    return None

def pairwise_dist(a,b):
    dx=a[0]-b[0]; dy=a[1]-b[1]
    return math.hypot(dx,dy)

net = jetson.inference.poseNet(args.model, sys.argv, args.threshold)
camera = jetson.utils.videoSource(args.source)
display = jetson.utils.videoOutput(args.output)
font = jetson.utils.cudaFont()
name_to_id = build_keypoint_index_map(net)

writer=None
if args.csv:
    os.makedirs(os.path.dirname(args.csv), exist_ok=True) if os.path.dirname(args.csv) else None
    csvfile=open(args.csv,"w",newline="")
    writer=csv.writer(csvfile)
    writer.writerow(["timestamp","num_people","pair_i","pair_j","pixel_distance","approx_meters","violation"])

last_time=time.time()

while display.IsStreaming():
    img = camera.Capture()
    poses = net.Process(img)
    if not args.no_skeleton:
        net.Overlay(img,"links,keypoints")

    centers=[]
    for idx,p in enumerate(poses):
        c = hip_center_for_pose(p,name_to_id)
        if c:
            centers.append((idx,c))
            jetson.utils.cudaDrawCircle(img,(int(c[0]),int(c[1])),8,(255,255,255,255))

    violations=[]
    for i in range(len(centers)):
        for j in range(i+1,len(centers)):
            pi,ci = centers[i]
            pj,cj = centers[j]
            d = pairwise_dist(ci,cj)
            bad = d < args.dist_thresh
            color = (255,0,0,255) if bad else (0,255,0,80)
            jetson.utils.cudaDrawLine(img,(int(ci[0]),int(ci[1])),(int(cj[0]),int(cj[1])),color,3 if bad else 1)

            midx,midy = int((ci[0]+cj[0])/2), int((ci[1]+cj[1])/2)
            label = f"{d:.0f}px"
            if args.ppm>0:
                meters = d/args.ppm
                label += f" ({meters:.2f}m)"
            font.OverlayText(img,img.width,img.height,label,midx,midy,(255,255,0,255),(0,0,0,160))

            if bad: violations.append((pi,pj,d))
            if writer:
                meters = (d/args.ppm) if args.ppm>0 else 0.0
                writer.writerow([time.time(), len(poses), pi, pj, f"{d:.3f}", f"{meters:.3f}", int(bad)])

    color = (255,50,50,255) if violations else (50,255,120,255)
    status = f"People: {len(poses)} | Violations: {len(violations)} | thresh={args.dist_thresh:.0f}px"
    font.OverlayText(img,img.width,img.height,status,12,10,color,(0,0,0,160))

    if args.max_fps>0:
        now=time.time(); dt=now-last_time; min_dt=1.0/args.max_fps
        if dt<min_dt: time.sleep(min_dt-dt)
        last_time=time.time()

    display.Render(img)
    display.SetStatus(status)

try:
    csvfile.close()
except Exception:
    pass
