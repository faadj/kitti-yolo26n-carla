"""
HW5 - CARLA Manual Driving with YOLO26n Live Object Detection
=============================================================
Uses ONNX runtime — works with Python 3.7 and CARLA 0.9.11
Low settings mode — reduced resolution, fewer NPCs, lower GPU load

Controls:
    W / Up Arrow    : Throttle
    S / Down Arrow  : Brake
    A / Left Arrow  : Steer left
    D / Right Arrow : Steer right
    Space           : Handbrake
    Q               : Toggle reverse
    R               : Start / Stop recording
    ESC             : Quit
"""

import sys
import os
import time
import random
import datetime
import numpy as np
import cv2
import onnxruntime as ort

# CARLA 0.9.11 Python API
sys.path.append('/home/fahad/carla_0.9.11/PythonAPI/carla/dist/carla-0.9.11-py3.7-linux-x86_64.egg')
sys.path.append('/home/fahad/carla_0.9.11/PythonAPI/carla')
import carla

# ── Configuration ─────────────────────────────────────────────────────────────

MODEL_PATH   = "/home/fahad/carla_0.9.11/kitti_yolo26n/best.onnx"
HOST         = "localhost"
PORT         = 2000
TOWN         = "Town03"
CAM_WIDTH    = 800               # reduced from 1280
CAM_HEIGHT   = 600               # reduced from 720
CAM_FOV      = 90
CONF_THRESH  = 0.35
NPC_VEHICLES = 15                # reduced from 30
NPC_WALKERS  = 10                # reduced from 20
RECORD_DIR   = "/home/fahad/carla_recordings"

# KITTI class names — must match kitti.yaml order
CLASS_NAMES = ["car", "van", "truck", "pedestrian",
               "person_sitting", "cyclist", "tram", "misc"]

# BGR colours per class
CLASS_COLORS = {
    "car":            (0,   200, 0  ),
    "van":            (0,   150, 255),
    "truck":          (0,   0,   255),
    "pedestrian":     (255, 50,  50 ),
    "person_sitting": (255, 150, 50 ),
    "cyclist":        (255, 255, 0  ),
    "tram":           (200, 0,   200),
    "misc":           (150, 150, 150),
}

# ── Globals ───────────────────────────────────────────────────────────────────

latest_frame = None
reverse_mode = False
is_recording = False
video_writer = None
record_path  = None

# ── Load ONNX model ───────────────────────────────────────────────────────────

print("[INFO] Loading YOLO26n ONNX model...")
session     = ort.InferenceSession(MODEL_PATH)
input_name  = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
print(f"[INFO] Model loaded. Input: {session.get_inputs()[0].shape}")

# ── ONNX inference ────────────────────────────────────────────────────────────

def preprocess(frame):
    """Resize and normalise frame for YOLO26n input."""
    img = cv2.resize(frame, (640, 640))
    img = img[:, :, ::-1]               # BGR → RGB
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC → CHW
    img = np.expand_dims(img, 0)        # → (1, 3, 640, 640)
    return img

def postprocess(outputs, orig_w, orig_h, conf_thresh):
    """
    Parse YOLO26n ONNX output shape (1, 300, 6).
    Each row: [x1, y1, x2, y2, confidence, class_id]
    Coordinates are in 640x640 space — scale back to original.
    """
    detections = outputs[0]   # (300, 6)
    results    = []
    scale_x    = orig_w / 640.0
    scale_y    = orig_h / 640.0
    for det in detections:
        x1, y1, x2, y2, conf, cls_id = det
        if conf < conf_thresh:
            continue
        x1 = int(x1 * scale_x)
        y1 = int(y1 * scale_y)
        x2 = int(x2 * scale_x)
        y2 = int(y2 * scale_y)
        results.append((x1, y1, x2, y2, float(conf), int(cls_id)))
    return results

def run_detection(frame):
    """Run ONNX inference on a BGR frame. Returns list of detections."""
    h, w = frame.shape[:2]
    inp  = preprocess(frame)
    out  = session.run([output_name], {input_name: inp})
    return postprocess(out[0], w, h, CONF_THRESH)

# ── Draw detections ───────────────────────────────────────────────────────────

def draw_detections(frame, detections):
    for (x1, y1, x2, y2, conf, cls_id) in detections:
        cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else "unknown"
        color    = CLASS_COLORS.get(cls_name, (200, 200, 200))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label       = f"{cls_name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return frame

# ── HUD ───────────────────────────────────────────────────────────────────────

def draw_hud(frame, speed_kmh, reverse, recording):
    h, w  = frame.shape[:2]
    gear  = "R" if reverse else "D"
    color = (0, 50, 255) if reverse else (0, 220, 0)
    cv2.putText(frame, f"{speed_kmh:.0f} km/h  [{gear}]",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
    if recording:
        cv2.circle(frame, (w - 30, 30), 10, (0, 0, 255), -1)
        cv2.putText(frame, "REC", (w - 75, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "YOLO26n ONNX | KITTI",
                (20, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, "W/S:Drive  A/D:Steer  Q:Reverse  R:Record  Space:Brake  ESC:Quit",
                (20, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)
    return frame

# ── Recording ─────────────────────────────────────────────────────────────────

def start_recording():
    global is_recording, video_writer, record_path
    os.makedirs(RECORD_DIR, exist_ok=True)
    ts           = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    record_path  = os.path.join(RECORD_DIR, f"carla_yolo26n_{ts}.avi")
    fourcc       = cv2.VideoWriter_fourcc(*"XVID")
    video_writer = cv2.VideoWriter(record_path, fourcc, 20.0, (CAM_WIDTH, CAM_HEIGHT))
    is_recording = True
    print(f"[REC] Started → {record_path}")

def stop_recording():
    global is_recording, video_writer
    if video_writer:
        video_writer.release()
        video_writer = None
    is_recording = False
    print(f"[REC] Saved → {record_path}")

# ── Camera callback ───────────────────────────────────────────────────────────

def camera_callback(image):
    global latest_frame
    array        = np.frombuffer(image.raw_data, dtype=np.uint8)
    array        = array.reshape((image.height, image.width, 4))
    latest_frame = array[:, :, :3].copy()

# ── Keyboard → vehicle control ────────────────────────────────────────────────

def parse_keys(keys):
    global reverse_mode
    control            = carla.VehicleControl()
    control.hand_brake = False
    if keys[ord('q')]:
        reverse_mode = not reverse_mode
        time.sleep(0.2)
    control.reverse = reverse_mode
    if keys[ord('w')] or keys[82]:  control.throttle = 0.7
    if keys[ord('s')] or keys[84]:  control.brake    = 0.8
    if keys[ord('a')] or keys[81]:  control.steer    = -0.5
    if keys[ord('d')] or keys[83]:  control.steer    =  0.5
    if keys[ord(' ')]:              control.hand_brake = True
    return control

# ── Spawn NPC traffic ─────────────────────────────────────────────────────────

def spawn_npc(world, traffic_manager, n_vehicles, n_walkers):
    bp_lib       = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)
    vehicle_actors = []
    walker_actors  = []

    vehicle_bps = bp_lib.filter("vehicle.*")
    for sp in spawn_points[:n_vehicles]:
        bp = random.choice(vehicle_bps)
        try:
            v = world.spawn_actor(bp, sp)
            v.set_autopilot(True, traffic_manager.get_port())
            vehicle_actors.append(v)
        except Exception:
            pass

    walker_bps     = [bp for bp in bp_lib.filter("walker.pedestrian.*")]
    walker_ctrl_bp = bp_lib.find("controller.ai.walker")
    for _ in range(n_walkers):
        sp = carla.Transform(world.get_random_location_from_navigation())
        bp = random.choice(walker_bps)
        if bp.has_attribute("is_invincible"):
            bp.set_attribute("is_invincible", "false")
        try:
            walker = world.spawn_actor(bp, sp)
            ctrl   = world.spawn_actor(walker_ctrl_bp, carla.Transform(), attach_to=walker)
            world.tick()
            ctrl.start()
            ctrl.go_to_location(world.get_random_location_from_navigation())
            ctrl.set_max_speed(1.4)
            walker_actors.append((walker, ctrl))
        except Exception:
            pass

    print(f"[INFO] Spawned {len(vehicle_actors)} vehicles, {len(walker_actors)} walkers.")
    return vehicle_actors, walker_actors

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    global latest_frame, reverse_mode, is_recording, video_writer, record_path

    client         = None
    ego_vehicle    = None
    camera_sensor  = None
    vehicle_actors = []
    walker_actors  = []
    prev_r         = False

    try:
        print(f"[INFO] Connecting to CARLA at {HOST}:{PORT} ...")
        client = carla.Client(HOST, PORT)
        client.set_timeout(20.0)
        world  = client.load_world(TOWN)
        print(f"[INFO] Loaded {TOWN}.")

        # Apply low rendering settings to reduce GPU load
        world_settings = world.get_settings()
        world_settings.synchronous_mode    = True
        world_settings.fixed_delta_seconds = 0.05
        world_settings.no_rendering_mode   = False  # keep rendering on for camera
        world.apply_settings(world_settings)

        # Set weather to clear noon — lightest rendering load
        world.set_weather(carla.WeatherParameters.ClearNoon)

        traffic_manager = client.get_trafficmanager(8000)
        traffic_manager.set_synchronous_mode(True)
        traffic_manager.set_global_distance_to_leading_vehicle(2.0)

        bp_lib      = world.get_blueprint_library()
        car_bp      = bp_lib.find("vehicle.tesla.model3")
        spawn_pts   = world.get_map().get_spawn_points()
        ego_vehicle = world.spawn_actor(car_bp, random.choice(spawn_pts))
        print("[INFO] Player vehicle spawned.")

        # Camera at reduced resolution for lower GPU load
        cam_bp = bp_lib.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(CAM_WIDTH))
        cam_bp.set_attribute("image_size_y", str(CAM_HEIGHT))
        cam_bp.set_attribute("fov",          str(CAM_FOV))
        cam_bp.set_attribute("sensor_tick",  "0.05")  # 20 FPS max
        cam_tf = carla.Transform(carla.Location(x=2.0, z=1.4),
                                 carla.Rotation(pitch=-5))
        camera_sensor = world.spawn_actor(cam_bp, cam_tf, attach_to=ego_vehicle)
        camera_sensor.listen(camera_callback)
        print("[INFO] Camera attached at 800x600.")

        vehicle_actors, walker_actors = spawn_npc(
            world, traffic_manager, NPC_VEHICLES, NPC_WALKERS)

        cv2.namedWindow("CARLA — YOLO26n Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("CARLA — YOLO26n Detection", CAM_WIDTH, CAM_HEIGHT)
        print("[INFO] Running. W/A/S/D to drive, R to record, ESC to quit.")

        while True:
            world.tick()

            raw_key = cv2.waitKey(1) & 0xFF
            keys    = np.zeros(256, dtype=np.uint8)
            if raw_key != 255:
                keys[raw_key] = 1

            if keys[27]:  # ESC
                break

            r_now = bool(keys[ord('r')])
            if r_now and not prev_r:
                if not is_recording:
                    start_recording()
                else:
                    stop_recording()
            prev_r = r_now

            ego_vehicle.apply_control(parse_keys(keys))

            vel       = ego_vehicle.get_velocity()
            speed_kmh = 3.6 * (vel.x**2 + vel.y**2 + vel.z**2) ** 0.5

            if latest_frame is not None:
                frame      = latest_frame.copy()
                detections = run_detection(frame)
                frame      = draw_detections(frame, detections)
                frame      = draw_hud(frame, speed_kmh, reverse_mode, is_recording)
                cv2.imshow("CARLA — YOLO26n Detection", frame)
                if is_recording and video_writer is not None:
                    video_writer.write(frame)

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted.")

    finally:
        print("[INFO] Cleaning up...")
        if is_recording:
            stop_recording()
        if client is not None:
            try:
                world    = client.get_world()
                settings = world.get_settings()
                settings.synchronous_mode    = False
                settings.fixed_delta_seconds = None
                world.apply_settings(settings)
            except Exception:
                pass
        for walker, ctrl in walker_actors:
            try: ctrl.stop(); ctrl.destroy(); walker.destroy()
            except Exception: pass
        for v in vehicle_actors:
            try: v.destroy()
            except Exception: pass
        if camera_sensor:
            try: camera_sensor.destroy()
            except Exception: pass
        if ego_vehicle:
            try: ego_vehicle.destroy()
            except Exception: pass
        cv2.destroyAllWindows()
        print("[INFO] Done.")

if __name__ == "__main__":
    main()
