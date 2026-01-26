import cv2
import numpy as np

cap = cv2.VideoCapture("bucket.mp4")

# -----------------------------
# Temporal smoothing variables
# -----------------------------
prev_cx, prev_cy = None, None
last_good_ellipse = None

ALPHA = 0.7        # smoothing factor
MAX_JUMP = 40      # reject sudden jumps

# -----------------------------
# Main loop
# -----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    img_cx, img_cy = w // 2, h // 2

    # -----------------------------
    # Segmentation
    # -----------------------------
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    best_ellipse = None
    max_area = 0

    MIN_AREA = 2000
    MAX_AREA = 0.9 * h * w

    # -----------------------------
    # Find best ellipse
    # -----------------------------
    for cnt in contours:
        if len(cnt) < 20:
            continue

        cnt_area = cv2.contourArea(cnt)
        if cnt_area < MIN_AREA or cnt_area > MAX_AREA:
            continue

        try:
            ellipse = cv2.fitEllipse(cnt)
        except:
            continue

        (cx, cy), (A, B), angle = ellipse

        # --- Simple, safe eccentricity check
        a = max(A, B) / 2
        b = min(A, B) / 2
        eccentricity = np.sqrt(1 - (b * b) / (a * a))

        if eccentricity > 0.85:
            continue

        area = A * B
        if area > max_area:
            max_area = area
            best_ellipse = ellipse

    # -----------------------------
    # Freeze if nothing valid
    # -----------------------------
    if best_ellipse is not None:
        last_good_ellipse = best_ellipse
    else:
        best_ellipse = last_good_ellipse

    # -----------------------------
    # Smooth center + output
    # -----------------------------
    if best_ellipse is not None:
        (cx, cy), (A, B), angle = best_ellipse

        # --- Temporal smoothing
        if prev_cx is None:
            smooth_cx, smooth_cy = cx, cy
        else:
            if abs(cx - prev_cx) > MAX_JUMP or abs(cy - prev_cy) > MAX_JUMP:
                cx, cy = prev_cx, prev_cy

            smooth_cx = ALPHA * prev_cx + (1 - ALPHA) * cx
            smooth_cy = ALPHA * prev_cy + (1 - ALPHA) * cy

        prev_cx, prev_cy = smooth_cx, smooth_cy
        cx, cy = int(smooth_cx), int(smooth_cy)

        dx = cx - img_cx
        dy = cy - img_cy

        # -----------------------------
        # Ellipse shrink (SAFE ADDITION)
        # -----------------------------
        SCALE = 0.9   # try 0.85–0.95

        scaled_ellipse = (
            (cx, cy),
            (A * SCALE, B * SCALE),
            angle
        )

        # -----------------------------
        # Visualization
        # -----------------------------
        cv2.ellipse(frame, scaled_ellipse, (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
        cv2.circle(frame, (img_cx, img_cy), 5, (255, 0, 0), -1)

        print(f"dx={dx}, dy={dy}")

    cv2.imshow("Ellipse Alignment (Stable + Scaled)", frame)
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
