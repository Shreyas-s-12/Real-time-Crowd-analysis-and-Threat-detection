"""
Webcam Diagnostic Test Script for Real-Time Crowd Analysis and Threat Detection
"""

import sys
import time
import cv2

def run_webcam_diagnostics():
    print("=" * 60)
    print("      WEBCAM & OPENCV BACKEND DIAGNOSTIC TOOL")
    print("=" * 60)
    print(f"Python Version : {sys.version.split()[0]}")
    print(f"OpenCV Version : {cv2.__version__}")
    print("-" * 60)

    backends = [
        (cv2.CAP_DSHOW, "CAP_DSHOW (DirectShow)"),
        (cv2.CAP_MSMF,  "CAP_MSMF (Media Foundation)"),
        (cv2.CAP_ANY,   "CAP_ANY (Default)")
    ]

    working_cameras = []

    for camera_id in range(4):
        print(f"\n--- Checking Camera Index {camera_id} ---")
        camera_found = False

        for backend_id, backend_name in backends:
            try:
                cap = cv2.VideoCapture(camera_id, backend_id) if backend_id is not None else cv2.VideoCapture(camera_id)
                opened = cap.isOpened() if cap else False

                if opened:
                    # Attempt warmup frame reads
                    frame_read_ok = False
                    frame_shape = None
                    for _ in range(5):
                        ret, frame = cap.read()
                        if ret and frame is not None and frame.size > 0:
                            frame_read_ok = True
                            frame_shape = frame.shape
                            break
                        time.sleep(0.05)

                    cap.release()

                    if frame_read_ok:
                        print(f"  [SUCCESS] Backend: {backend_name:30s} | Resolution: {frame_shape[1]}x{frame_shape[0]} | Channels: {frame_shape[2]}")
                        working_cameras.append({
                            'index': camera_id,
                            'backend': backend_name,
                            'resolution': (frame_shape[1], frame_shape[0])
                        })
                        camera_found = True
                        break
                    else:
                        print(f"  [FAILED ] Backend: {backend_name:30s} | Opened handle, but failed to read valid frames.")
                else:
                    if cap:
                        cap.release()
                    print(f"  [FAILED ] Backend: {backend_name:30s} | Could not open camera handle (isOpened=False).")
            except Exception as e:
                print(f"  [ERROR  ] Backend: {backend_name:30s} | Exception: {e}")

    print("\n" + "=" * 60)
    print("                   DIAGNOSTIC SUMMARY")
    print("=" * 60)
    if working_cameras:
        print(f"Found {len(working_cameras)} active camera device(s):")
        for cam in working_cameras:
            print(f" - Camera Index {cam['index']}: Works with {cam['backend']} at {cam['resolution'][0]}x{cam['resolution'][1]}")
        print("\nHardware and driver state: OK. Use index", working_cameras[0]['index'], "in the application.")
    else:
        print("CRITICAL: No working webcam devices were detected across indices 0-3.")
        print("Troubleshooting steps:")
        print(" 1. Check if the physical webcam is plugged in / enabled.")
        print(" 2. Verify Privacy & Security settings in Windows -> Camera permissions.")
        print(" 3. Ensure no other application (Zoom, Teams, Skype, Browser) is exclusively locking the webcam.")
    print("=" * 60)

if __name__ == "__main__":
    run_webcam_diagnostics()
