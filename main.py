import cv2
import numpy as np
import time
from gesture import GestureTracker
from effects import EFFECTS, apply_effect_in_quad
HELP_TEXT = "1-7: change effect | h: toggle help | q: quit"
def main():
    tracker = GestureTracker(model_path="hand_landmarker.task")
    current_effect_key = ord('1')
    show_help = True
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    print("Finger Frame Effect running. Press 'h' for on-screen help, 'q' to quit.")
    start_time = time.time()
    prev_frame_time = start_time
    fps = 0.0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        now = time.time()
        instant_fps = 1.0 / max(now - prev_frame_time, 1e-6)
        fps = 0.9 * fps + 0.1 * instant_fps
        prev_frame_time = now
        timestamp_ms = int((now - start_time) * 1000)
        quad = tracker.update(frame, timestamp_ms)
        if quad is not None:
            quad_int = quad.astype(np.int32)
            effect_name, effect_fn = EFFECTS[current_effect_key]
            frame = apply_effect_in_quad(frame, quad_int, effect_fn)
            cv2.polylines(frame, [quad_int], isClosed=True, color=(0, 255, 255), thickness=2)
            cv2.putText(frame, f"Effect: {effect_name}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Make an 'L' with BOTH hands to form a frame",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, frame.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        if show_help:
            cv2.putText(frame, HELP_TEXT, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        cv2.imshow("Finger Frame Effect", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('h'):
            show_help = not show_help
        elif key in EFFECTS:
            current_effect_key = key
    cap.release()
    cv2.destroyAllWindows()
    tracker.close()
if __name__ == "__main__":
    main()