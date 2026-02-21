import cv2
import numpy as np
import math

cap = cv2.VideoCapture("blue_orange_bucket.mp4")

if not cap.isOpened():
    print("Cannot open video")
    exit()

cv2.namedWindow("Frame")
cv2.namedWindow("Mask")

# ---------------- CAMERA CALIBRATION ----------------
camera_matrix = np.array([
    [620.31496735, 0, 325.04442149],
    [0, 616.05767137, 247.22452100],
    [0, 0, 1]
], dtype=np.float32)

dist_coeffs = np.array([
    [-0.43458364, 0.25736404, -0.00494952, 0.00179857, -0.28597311]
], dtype=np.float32)

# ---------------- BUCKET GEOMETRY ----------------
R = 0.235

object_points = np.array([
    (0, -R, 0),
    (R, 0, 0),
    (0, R, 0),
    (-R, 0, 0)
], dtype=np.float32)

def get_ellipse_rim_points(cx, cy, A, B, angle_deg):
    angle = np.deg2rad(angle_deg)
    a = A / 2
    b = B / 2

    cos_t = np.cos(angle)
    sin_t = np.sin(angle)

    ux = cos_t
    uy = sin_t
    vx = -sin_t
    vy = cos_t

    right = (cx + a * ux, cy + a * uy)
    left  = (cx - a * ux, cy - a * uy)
    top   = (cx + b * vx, cy + b * vy)
    bottom= (cx - b * vx, cy - b * vy)

    return np.array([top, right, bottom, left], dtype=np.float32)

MIN_AREA_RATIO = 0.01
MAX_AREA_RATIO = 0.85
EDGE_MARGIN = 40

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    blur = cv2.GaussianBlur(hsv, (5, 5), 0)

    # ---------------- BLUE MASK ----------------
    lower_blue = np.array([100, 80, 50])
    upper_blue = np.array([140, 255, 255])
    mask_blue = cv2.inRange(blur, lower_blue, upper_blue)

    # ---------------- ORANGE MASK ----------------
    lower_orange = np.array([0, 40, 40])
    upper_orange = np.array([45, 255, 255])
    mask_orange = cv2.inRange(blur, lower_orange, upper_orange)

    mask = cv2.bitwise_or(mask_blue, mask_orange)

    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    cv2.imshow("Mask", mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    MIN_AREA = MIN_AREA_RATIO * h * w
    MAX_AREA = MAX_AREA_RATIO * h * w

    best_area = 0
    best_ellipse = None
    detected_color = None

    for cnt in contours:
        if len(cnt) < 40:
            continue

        area = cv2.contourArea(cnt)
        if area < MIN_AREA or area > MAX_AREA:
            continue

        try:
            ellipse = cv2.fitEllipse(cnt)
        except:
            continue

        (cx, cy), (A, B), angle = ellipse

        if cx < EDGE_MARGIN or cy < EDGE_MARGIN or cx > w - EDGE_MARGIN or cy > h - EDGE_MARGIN:
            continue

        if area > best_area:
            best_area = area
            best_ellipse = ellipse

            mask_roi = np.zeros_like(mask)
            cv2.drawContours(mask_roi, [cnt], -1, 255, -1)

            blue_pixels = cv2.countNonZero(cv2.bitwise_and(mask_blue, mask_roi))
            orange_pixels = cv2.countNonZero(cv2.bitwise_and(mask_orange, mask_roi))

            if blue_pixels > orange_pixels:
                detected_color = "Blue Bucket"
            else:
                detected_color = "Orange Bucket"

    if best_ellipse is not None:
        (cx, cy), (A, B), angle = best_ellipse

        cv2.ellipse(frame, best_ellipse, (0, 255, 0), 3)

        # -------- ADD TVEC PART ONLY --------
        image_points = get_ellipse_rim_points(cx, cy, A, B, angle)

        success, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE
        )

        if success:
            tx, ty, tz = tvec.flatten()
            print(f"{detected_color} | tx={tx:.3f}  ty={ty:.3f}  tz={tz:.3f}")

            cv2.putText(frame,
                        f"tz = {tz:.2f} m",
                        (50, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2)

        if detected_color:
            cv2.putText(frame,
                        detected_color,
                        (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2)

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
