import cv2
import mediapipe as mp
import numpy as np
import datetime

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)

class PushUpCounter:
    def _init_(self, weight=70):
        self.push_up_count = 0
        self.is_paused = False
        self.start_y = None
        self.going_down = False
        self.session_start_time = datetime.datetime.now()
        self.calories_burned = 0
        self.weight = weight
        self.CALORIES_PER_PUSHUP = 0.29 * (self.weight / 70)
        self.target_pushups = 20
        self.form_feedback = "Form: Perfect!"

    def check_form(self, landmarks):
        shoulder = [
            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
        ]
        elbow = [
            landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y
        ]
        wrist = [
            landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y
        ]

        angle = self.calculate_angle(shoulder, elbow, wrist)

        if angle < 70:
            self.form_feedback = "Form: Terlalu Rendah!"
            return (0, 0, 255)  # Red
        elif angle > 110:
            self.form_feedback = "Form: Terlalu Tinggi!"
            return (0, 165, 255)  # Orange
        else:
            self.form_feedback = "Form: Perfect!"
            return (0, 255, 0)  # Green

    def calculate_angle(self, a, b, c):
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180.0:
            angle = 360 - angle
        return angle

    def process_frame(self, frame):
        if not self.is_paused:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)

            if results.pose_landmarks:
                # Draw pose landmarks with white color
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
                )

                head_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y
                if self.start_y is None:
                    self.start_y = head_y

                form_color = self.check_form(results.pose_landmarks.landmark)

                if head_y > self.start_y + 0.05 and not self.going_down:
                    self.going_down = True
                elif head_y < self.start_y and self.going_down:
                    self.push_up_count += 1
                    self.going_down = False
                    self.calories_burned = self.push_up_count * self.CALORIES_PER_PUSHUP

                self.display_info(frame, form_color)

        return frame

    def display_info(self, frame, form_color):
        # Buat background semi-transparan untuk area teks
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (10, 10), (350, 250),  # Area teks diperbesar untuk desain yang lebih lega
            (0, 0, 0),  # Warna hitam sebagai background
            -1
        )
        
        # Blend overlay dengan frame asli
        alpha = 0.6  # Transparansi lebih lembut
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Pengaturan font
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.9
        thickness = 2
        
        # Push-up count (Hijau untuk progress)
        cv2.putText(
            frame,
            f"Push-ups: {self.push_up_count}/{self.target_pushups}",
            (20, 50),
            font,
            font_scale,
            (0, 255, 0),  # Warna hijau
            thickness
        )

        # Progress Bar Hijau dengan gradasi
        progress_bar_width = int((self.push_up_count / self.target_pushups) * 300)
        for i in range(progress_bar_width):  # Buat gradasi hijau
            color = (0, 255 - int((i / 300) * 255), int((i / 300) * 255))
            cv2.line(frame, (20 + i, 70), (20 + i, 90), color, 2)

        # Garis tepi progress bar
        cv2.rectangle(
            frame,
            (20, 70), (320, 90),  # Batas progress bar maksimal
            (255, 255, 255),  # Warna putih
            2  # Ketebalan garis
        )
        
        # Form feedback (warna dinamis berdasarkan form)
        cv2.putText(
            frame,
            f"Form: {self.form_feedback.split(': ')[1]}",
            (20, 120),
            font,
            font_scale,
            form_color,
            thickness
        )
        
        # Kalori (warna biru muda untuk kontras)
        cv2.putText(
            frame,
            f"Calories: {self.calories_burned:.1f}",
            (20, 160),
            font,
            font_scale,
            (255, 204, 0),  # Warna oranye keemasan
            thickness
        )
        
        # Waktu sesi (warna putih lebih kecil)
        session_duration = datetime.datetime.now() - self.session_start_time
        time_str = str(session_duration).split('.')[0]
        cv2.putText(
            frame,
            f"Time: {time_str}",
            (20, 200),
            font,
            font_scale - 0.2,  # Ukuran font lebih kecil
            (255, 255, 255),  # Warna putih
            thickness
        )

        # Kontrol teks di bawah (putih dengan padding lebih besar)
        controls_text = "SPACE: Pause/Resume | Q: Quit | R: Reset"
        cv2.putText(
            frame,
            controls_text,
            (20, frame.shape[0] - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 200, 200),
            1
        )

def main():
    # Menggunakan video file
    cap = cv2.VideoCapture('pusssupp.mp4')
    counter = PushUpCounter(weight=70)

    cv2.namedWindow("Push-up Counter Pro", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Push-up Counter Pro", 1280, 720)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = counter.process_frame(frame)
        cv2.imshow("Push-up Counter Pro", processed_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            counter.is_paused = not counter.is_paused
        elif key == ord('r'):
            counter = PushUpCounter(weight=70)
        elif key == ord('t'):
            try:
                target = int(input("Enter target pushups: "))
                if target > 0:
                    counter.target_pushups = target
            except ValueError:
                print("Please enter a valid number")

    cap.release()
    cv2.destroyAllWindows()

if _name_ == "_main_":
    main()
