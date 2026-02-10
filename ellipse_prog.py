import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

VIDEO_PATH = "bucket.mp4"
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    exit(1)

ret, _ = cap.read()
if not ret:
    exit(1)

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

prev_cx, prev_cy = None, None
last_good_center = None

ALPHA = 0.6
MAX_JUMP = 40
GOOD_FRAMES_REQUIRED = 5
BAD_FRAMES_LIMIT = 8

good_count = 0
bad_count = 0

MIN_AREA_RATIO = 0.01
MAX_AREA_RATIO = 0.6
EDGE_MARGIN = 40
RIM_BAND_RATIO = 0.12

plt.ion()
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 5))
dx_vals, dy_vals = [], []

line_dx, = ax1.plot([], [], 'r')
line_dy, = ax2.plot([], [], 'b')

ax1.set_ylabel("dx")
ax2.set_ylabel("dy")
ax2.set_xlabel("Frame")

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    img_cx, img_cy = w // 2, h // 2

    # gray = cv2.cvtColor(frame, cv2.COLORBGR2HSV)
    # blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # _, thresh = cv2.threshold(
    #     blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    # )

    # contours, _ = cv2.findContours(
    #     thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    # )

    # Convert BGR → HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Convert to GRAYSCALE (not HSV!)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Blur to reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Otsu's binarization
    _, thresh = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Show Otsu mask
    cv2.imshow("Otsu Mask", thresh)
    # Blur to reduce noise
    blur = cv2.GaussianBlur(hsv, (5, 5), 0)

    # Define BLUE color range in HSV
    lower_blue = np.array([100, 80, 50])
    upper_blue = np.array([140, 255, 255])

    # Create mask for blue color
    mask = cv2.inRange(blur, lower_blue, upper_blue)

    # (Optional) clean the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    masked_result = cv2.bitwise_and(frame, frame, mask=mask)
    cv2.imshow("blue masked result", masked_result)

    # Find contours on the MASKED image
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )


    best_center = None
    best_radius = None
    best_score = 0

    MIN_AREA = MIN_AREA_RATIO * h * w
    MAX_AREA = MAX_AREA_RATIO * h * w

    for cnt in contours:
        if len(cnt) < 40:
            continue

        cnt_area = cv2.contourArea(cnt)
        if cnt_area < MIN_AREA or cnt_area > MAX_AREA:
            continue

        try:
            (cx, cy), (A, B), _ = cv2.fitEllipse(cnt)
        except:
            continue

        if cx < EDGE_MARGIN or cy < EDGE_MARGIN or cx > w-EDGE_MARGIN or cy > h-EDGE_MARGIN:
            continue

        a = max(A, B) / 2
        b = min(A, B) / 2
        if a <= 0 or b <= 0:
            continue

        rim_points = []
        inner_limit = 1.0 - RIM_BAND_RATIO

        for p in cnt:
            px, py = p[0]
            dxn = (px - cx) / a
            dyn = (py - cy) / b
            r = math.sqrt(dxn*dxn + dyn*dyn)
            if inner_limit < r <= 1.05:
                rim_points.append([px, py])

        if len(rim_points) < 30:
            continue

        rim_points = np.array(rim_points).reshape(-1, 1, 2)

        try:
            (cx2, cy2), (A2, B2), _ = cv2.fitEllipse(rim_points)
        except:
            continue

        if max(A2, B2) / min(A2, B2) > 1.6:
            continue

        score = cv2.contourArea(rim_points)
        if score > best_score:
            best_score = score
            best_center = (cx2, cy2)
            best_radius = int(0.95 * math.sqrt((A2 / 2) * (B2 / 2)))

    if best_center is not None:
        good_count += 1
        bad_count = 0
    else:
        bad_count += 1
        good_count = 0

    if good_count >= GOOD_FRAMES_REQUIRED:
        last_good_center = best_center

    if bad_count >= BAD_FRAMES_LIMIT:
        last_good_center = None
        prev_cx, prev_cy = None, None

    if last_good_center is not None:
        cx, cy = last_good_center

        if prev_cx is not None and math.hypot(cx-prev_cx, cy-prev_cy) > MAX_JUMP:
            cx, cy = prev_cx, prev_cy

        if prev_cx is None:
            smooth_cx, smooth_cy = cx, cy
        else:
            smooth_cx = ALPHA * prev_cx + (1 - ALPHA) * cx
            smooth_cy = ALPHA * prev_cy + (1 - ALPHA) * cy

        prev_cx, prev_cy = smooth_cx, smooth_cy
        cx, cy = int(smooth_cx), int(smooth_cy)

        dx = cx - img_cx
        dy = cy - img_cy

        print(f"dx={dx}, dy={dy}")

        dx_vals.append(dx)
        dy_vals.append(dy)

        if best_radius:
            cv2.circle(frame, (cx, cy), best_radius, (0, 255, 0), 2)

        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
        cv2.circle(frame, (img_cx, img_cy), 5, (255, 0, 0), -1)

    if frame_count % 5 == 0 and dx_vals:
        line_dx.set_data(range(len(dx_vals)), dx_vals)
        line_dy.set_data(range(len(dy_vals)), dy_vals)
        ax1.relim(); ax1.autoscale_view()
        ax2.relim(); ax2.autoscale_view()
        plt.pause(0.001)

    frame_count += 1
    cv2.imshow("Bucket Centering", frame)
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
plt.ioff()
plt.show()
  
      
