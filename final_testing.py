import cv2
import numpy as np
import math

cap = cv2.VideoCapture(0)  # live camera

if not cap.isOpened():
    print("Cannot open camera")
    exit()


# -------------------------------
# Rim point function
# -------------------------------
def get_ellipse_rim_points(cx, cy, A, B, angle_deg):
    angle = np.deg2rad(angle_deg)

    a = A / 2
    b = B / 2

    cos_t = np.cos(angle)
    sin_t = np.sin(angle)

    # Major axis direction
    ux = cos_t
    uy = sin_t

    # Minor axis direction
    vx = -sin_t
    vy = cos_t

    right = (cx + a * ux, cy + a * uy)
    left  = (cx - a * ux, cy - a * uy)
    top   = (cx + b * vx, cy + b * vy)
    bottom= (cx - b * vx, cy - b * vy)

    return np.array([top, left, bottom, right], dtype=np.float32)


# -------------------------------
# Camera matrix
# -------------------------------
# Calibrated camera matrix
    camera_matrix = np.array([
        [620.31496735, 0, 325.04442149],
        [0, 616.05767137, 247.22452100],
        [0, 0, 1]
    ], dtype=np.float32)

# Calibrated distortion coefficients
    dist_coeffs = np.array([
        [-0.43458364, 0.25736404, -0.00494952, 0.00179857, -0.28597311]
    ], dtype=np.float32)


# Detection parameters
MIN_AREA_RATIO = 0.01
MAX_AREA_RATIO = 0.6
EDGE_MARGIN = 40
RIM_BAND_RATIO = 0.12


while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    blur = cv2.GaussianBlur(hsv, (5, 5), 0)

    # Blue mask
    lower_blue = np.array([100, 80, 50])
    upper_blue = np.array([140, 255, 255])
    mask = cv2.inRange(blur, lower_blue, upper_blue)

    # Clean mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    MIN_AREA = MIN_AREA_RATIO * h * w
    MAX_AREA = MAX_AREA_RATIO * h * w

    best_score = 0
    best_ellipse = None

    for cnt in contours:
        if len(cnt) < 40:
            continue

        cnt_area = cv2.contourArea(cnt)
        if cnt_area < MIN_AREA or cnt_area > MAX_AREA:
            continue

        try:
            (cx, cy), (A, B), angle = cv2.fitEllipse(cnt)
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
            (cx2, cy2), (A2, B2), angle2 = cv2.fitEllipse(rim_points)
        except:
            continue

        if max(A2, B2) / min(A2, B2) > 1.6:
            continue

        score = cv2.contourArea(rim_points)
        if score > best_score:
            best_score = score
            best_ellipse = (cx2, cy2, A2, B2, angle2)





    # -------------------------------
    # If ellipse found, run solvePnP
    # -------------------------------
    if best_ellipse is not None:
        cx2, cy2, A2, B2, angle2 = best_ellipse

        # Draw ellipse
        cv2.ellipse(
            frame,
            ((cx2, cy2), (A2, B2), angle2),
            (0, 255, 0),
            2
        )

        # Get rim points
        image_points = get_ellipse_rim_points(cx2, cy2, A2, B2, angle2)

        # Draw rim points
        for (px, py) in image_points:
            cv2.circle(frame, (int(px), int(py)), 6, (0, 255, 255), -1)

        # Define object points (unit circle)
        object_points = np.array([
            (0, -0.235, 0),   # top
            (-0.235, 0, 0),   # left
            (0, 0.235, 0),    # bottom
            (0.235, 0, 0)     # right
        ], dtype=np.float32)

       

        success, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            camera_matrix,
            dist_coeffs
        )

        if success:
            tx, ty, tz = tvec.flatten()
            print(f"tvec: tx={tx:.3f}, ty={ty:.3f}, tz={tz:.3f}")

    cv2.imshow("Bucket solvePnP", frame)
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

