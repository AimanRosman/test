import cv2
import time
import requests
import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from ultralytics import YOLO

# --- 1. CONFIGURATION ---

# CHANGE THIS IP TO MATCH YOUR ESP32'S IP ADDRESS 
ESP32_URL = "http://192.168.0.148/update" 
MODEL_PATH = "model/best.pt" # Ensure this path is correct
SEND_INTERVAL = 1.0          # How often to send data to ESP32 (seconds)
SERVER_PORT = 8000
SERVER_URL = f"http://localhost:{SERVER_PORT}"
HTML_FILE_PATH = "index.html" # Path to the HTML file we are serving

# --- 2. GLOBAL STATE (Shared between threads) ---

current_ppe_status = {
    "hardhat": 1,
    "vest": 1,
    "mask": 1,
    "timestamp": time.time()
}

status_lock = threading.Lock()

# Global variable to store the latest frame
latest_frame = None
frame_lock = threading.Lock()

# --- 3. HTTP SERVER CLASS (Runs in a separate thread) ---

class DashboardHandler(BaseHTTPRequestHandler):
    """Handles requests for the HTML dashboard and the status JSON."""

    def do_GET(self):
        """Handle GET requests for /, /status, and /video_feed."""
        if self.path == '/':
            self._send_html()
        elif self.path == '/status':
            self._send_status_json()
        elif self.path == '/video_feed':
            self._send_video_feed()
        else:
            self._send_404()

    def _load_html_content(self):
        """Loads the content of the index.html file with UTF-8 encoding."""
        try:
            with open(HTML_FILE_PATH, 'r', encoding='utf-8') as f:
                return f.read().encode('utf-8')
        except FileNotFoundError:
            error_message = f"Error: {HTML_FILE_PATH} not found. Ensure it is in the same directory."
            print(f"[SERVER ERROR] {error_message}")
            return error_message.encode('utf-8')
        except Exception as e:
            error_message = f"Error reading {HTML_FILE_PATH}: {e}"
            print(f"[SERVER ERROR] {error_message}")
            return error_message.encode('utf-8')

    def _send_html(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html; charset=utf-8')
        self.end_headers()
        html_content = self._load_html_content()
        self.wfile.write(html_content)

    def _send_status_json(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        # Allow access from the local HTML page (CORS)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        
        # Access the global status variable safely
        with status_lock:
            response = current_ppe_status.copy()
        
        self.wfile.write(json.dumps(response).encode('utf-8'))

    def _send_video_feed(self):
        """Stream the video feed as MJPEG."""
        self.send_response(200)
        self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=frame')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        
        try:
            while True:
                with frame_lock:
                    if latest_frame is None:
                        time.sleep(0.1)
                        continue
                    frame = latest_frame.copy()
                
                # Encode frame as JPEG
                ret, buffer = cv2.imencode('.jpg', frame)
                if not ret:
                    continue
                
                frame_bytes = buffer.tobytes()
                
                # Send frame in multipart format
                self.wfile.write(b'--frame\r\n')
                self.send_header('Content-Type', 'image/jpeg')
                self.send_header('Content-Length', len(frame_bytes))
                self.end_headers()
                self.wfile.write(frame_bytes)
                self.wfile.write(b'\r\n')
                
                time.sleep(0.033)  # ~30 FPS
        except Exception as e:
            print(f"[VIDEO FEED ERROR] {e}")

    def _send_404(self):
        self.send_response(404)
        self.send_header('Content-type', 'text/plain')
        self.end_headers()
        self.wfile.write(b"404 Not Found")

    def log_message(self, format, *args):
        # Suppress logging every request to keep the console clean for CV output
        pass

def run_server():
    """Starts the simple HTTP server."""
    server_address = ('', SERVER_PORT)
    httpd = HTTPServer(server_address, DashboardHandler)
    print(f"[SERVER] Starting dashboard server at {SERVER_URL}...")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("[SERVER] Shutting down server...")
        httpd.server_close()

# --- 4. COMPUTER VISION LOOP (Main Thread) ---

def cv_loop():
    """Runs the YOLO model and updates the global status."""
    global current_ppe_status, latest_frame
    
    print("[YOLO] Initializing model and camera...")
    
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"[YOLO ERROR] Error loading model. Check MODEL_PATH: {e}")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[YOLO ERROR] Cannot open camera.")
        return

    last_send = 0
    print(f"[YOLO] PPE Detection running. Sending data to ESP32 ({ESP32_URL}) every {SEND_INTERVAL}s.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[YOLO] Camera error or stream ended")
                break

            # Run YOLO inference
            results = model(frame, verbose=False)
            
            # --- 4.1 Determine PPE Status ---
            ppe_status_local = {
                "hardhat": 1, "vest": 1, "mask": 1
            }

            annotated_frame = results[0].plot()

            for box in results[0].boxes:
                cls = int(box.cls[0])
                label = model.names[cls]

                # Set status to 0 (MISSING) if NO-PPE is detected
                if "NO-Hardhat" in label:
                    ppe_status_local["hardhat"] = 0
                if "NO-Safety Vest" in label:
                    ppe_status_local["vest"] = 0
                if "NO-Mask" in label:
                    ppe_status_local["mask"] = 0

            # --- 4.2 Update Global Status (for Web Display) ---
            with status_lock:
                current_ppe_status.update(ppe_status_local)
                current_ppe_status["timestamp"] = time.time()

            # Update latest frame for video streaming
            with frame_lock:
                latest_frame = annotated_frame.copy()

            # Show webcam window (Keep this for debugging CV)
            cv2.imshow("PPE Detection", annotated_frame)

            # --- 4.3 Send data to ESP32 (Physical Interface) ---
            current_time = time.time()
            if current_time - last_send > SEND_INTERVAL:
                try:
                    r = requests.post(ESP32_URL, json=ppe_status_local, timeout=0.5)
                    if r.status_code != 200:
                        print(f"[ESP32 WARNING] Response unexpected (Status: {r.status_code})")
                except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
                    print("[ESP32 ERROR] Could not connect to ESP32. Check IP/WiFi.")
                except Exception as e:
                    print(f"[ESP32 ERROR] Send error: {e}")

                last_send = current_time

            # Handle user exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # --- 4.4 Cleanup ---
        cap.release()
        cv2.destroyAllWindows()
        print("[YOLO] Application shut down.")


# --- 5. MAIN EXECUTION ---

if __name__ == '__main__':
    # Start the HTTP server in a background thread
    server_thread = threading.Thread(target=run_server)
    server_thread.daemon = True 
    server_thread.start()

    # Wait a moment for the server to spin up, then open the browser
    time.sleep(1) 
    print(f"[BROWSER] Opening dashboard in web browser: {SERVER_URL}")
    webbrowser.open(SERVER_URL)
    
    # Run the Computer Vision loop in the main thread
    cv_loop()
    
    # Wait for the server thread to clean up before exiting
    server_thread.join(timeout=1)
