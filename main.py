import tkinter as tk
from tkinter import font as tkfont
import threading
import time
import cv2
from PIL import Image, ImageTk
from face_handler import FaceMatcher
from interaction_store import InteractionStore
from summary_generator import SummaryGenerator
from transcribe_handler import TranscribeHandler
from dotenv import load_dotenv
import os

load_dotenv()

COLLECTION_ID = os.getenv("REKOGNITION_COLLECTION", "memoire-faces")
BUCKET_NAME   = os.getenv("S3_BUCKET", "memoire-faces-yjt")
TABLE_NAME    = os.getenv("DYNAMODB_TABLE", "interactions")
CONFIDENCE    = float(os.getenv("CONFIDENCE_THRESHOLD", "80"))


class MemoireApp:
    def __init__(self, root):
        self.root = root
        self.root.title("MemoireAR")
        self.root.configure(bg="#0a0a0f")
        self.root.geometry("1100x720")
        self.root.resizable(False, False)

        self.face_matcher     = FaceMatcher(COLLECTION_ID, BUCKET_NAME, CONFIDENCE)
        self.interaction_store = InteractionStore(TABLE_NAME)
        self.summary_gen      = SummaryGenerator()
        self.transcriber      = TranscribeHandler()

        self.current_person   = None
        self.running          = True
        self.last_match_time  = 0
        self.cooldown         = 30        # only scan every 30 seconds
        self.is_recording     = False
        self.has_recognized   = False     # lock after first recognition

        self._build_ui()
        self._start_camera()

    # ── UI ───────────────────────────────────────────────
    def _build_ui(self):
        title_font  = tkfont.Font(family="Courier New", size=22, weight="bold")
        label_font  = tkfont.Font(family="Courier New", size=11, weight="bold")
        body_font   = tkfont.Font(family="Courier New", size=11)
        status_font = tkfont.Font(family="Courier New", size=10)

        # header
        header = tk.Frame(self.root, bg="#0a0a0f")
        header.pack(fill="x", padx=24, pady=(18, 0))
        tk.Label(header, text="◈ MEMOIRE", font=title_font,
                 fg="#c8a96e", bg="#0a0a0f").pack(side="left")
        tk.Label(header, text="dementia face recognition assistant",
                 font=status_font, fg="#444455", bg="#0a0a0f").pack(side="left", padx=14)
        self.status_dot = tk.Label(header, text="● SCANNING",
                                   font=status_font, fg="#3ddc84", bg="#0a0a0f")
        self.status_dot.pack(side="right")

        tk.Frame(self.root, bg="#222233", height=1).pack(fill="x", padx=24, pady=10)

        # columns
        cols = tk.Frame(self.root, bg="#0a0a0f")
        cols.pack(fill="both", expand=True, padx=24)

        # LEFT — camera
        left = tk.Frame(cols, bg="#0a0a0f")
        left.pack(side="left", fill="y")

        tk.Label(left, text="LIVE FEED", font=label_font,
                 fg="#555566", bg="#0a0a0f").pack(anchor="w", pady=(0, 6))

        cam_border = tk.Frame(left, bg="#222233", padx=2, pady=2)
        cam_border.pack()
        self.cam_label = tk.Label(cam_border, bg="#111120")
        self.cam_label.pack()

        self.person_badge = tk.Label(left, text="NO FACE DETECTED",
                                     font=label_font, fg="#444455", bg="#0a0a0f", pady=8)
        self.person_badge.pack(anchor="w", pady=(10, 0))

        # record button
        self.record_btn = tk.Button(
            left, text="⏺  START RECORDING",
            font=label_font, fg="#0a0a0f", bg="#555566",
            relief="flat", padx=14, pady=8,
            state="disabled",
            command=self._toggle_recording
        )
        self.record_btn.pack(fill="x", pady=(8, 0))

        # recording status
        self.rec_status = tk.Label(left, text="",
                                   font=status_font, fg="#ff4444", bg="#0a0a0f")
        self.rec_status.pack(anchor="w", pady=(4, 0))

        # RIGHT — panels
        right = tk.Frame(cols, bg="#0a0a0f")
        right.pack(side="right", fill="both", expand=True, padx=(24, 0))

        # live transcript
        tk.Label(right, text="LIVE TRANSCRIPT", font=label_font,
                 fg="#555566", bg="#0a0a0f").pack(anchor="w")
        trans_frame = tk.Frame(right, bg="#0d0d1a",
                               highlightbackground="#222233", highlightthickness=1)
        trans_frame.pack(fill="x", pady=(4, 16))
        self.transcript_text = tk.Text(trans_frame, height=4, font=body_font,
                                       fg="#8888aa", bg="#0d0d1a",
                                       wrap="word", relief="flat",
                                       padx=12, pady=10,
                                       state="disabled", cursor="arrow")
        self.transcript_text.pack(fill="x")

        # summary
        tk.Label(right, text="AI SUMMARY", font=label_font,
                 fg="#555566", bg="#0a0a0f").pack(anchor="w")
        sum_frame = tk.Frame(right, bg="#0d0d1a",
                             highlightbackground="#222233", highlightthickness=1)
        sum_frame.pack(fill="x", pady=(4, 16))
        self.summary_text = tk.Text(sum_frame, height=5, font=body_font,
                                    fg="#e8e0d0", bg="#0d0d1a",
                                    wrap="word", relief="flat",
                                    padx=12, pady=10,
                                    state="disabled", cursor="arrow")
        self.summary_text.pack(fill="x")

        # interaction log
        tk.Label(right, text="INTERACTION LOG", font=label_font,
                 fg="#555566", bg="#0a0a0f").pack(anchor="w")
        log_frame = tk.Frame(right, bg="#0d0d1a",
                             highlightbackground="#222233", highlightthickness=1)
        log_frame.pack(fill="both", expand=True, pady=(4, 0))
        self.log_text = tk.Text(log_frame, font=status_font,
                                fg="#8888aa", bg="#0d0d1a",
                                wrap="word", relief="flat",
                                padx=12, pady=10,
                                state="disabled", cursor="arrow")
        self.log_text.pack(fill="both", expand=True)

        # status bar
        tk.Frame(self.root, bg="#222233", height=1).pack(fill="x", padx=24, pady=(8, 0))
        self.bottom_status = tk.Label(self.root,
                                      text="System ready. Point camera at a known face.",
                                      font=status_font, fg="#444455", bg="#0a0a0f")
        self.bottom_status.pack(anchor="w", padx=24, pady=6)

    # ── CAMERA ──────────────────────────────────────────
    def _start_camera(self):
        self.cap = cv2.VideoCapture(0)
        self._update_frame()

    def _update_frame(self):
        if not self.running:
            return
        ret, frame = self.cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb).resize((480, 360))
            self.current_frame = frame
            imgtk = ImageTk.PhotoImage(img)
            self.cam_label.configure(image=imgtk)
            self.cam_label.image = imgtk

            if (time.time() - self.last_match_time > self.cooldown
                    and not self.is_recording
                    and not self.has_recognized):
                self.last_match_time = time.time()
                threading.Thread(target=self._scan_face,
                                 args=(frame,), daemon=True).start()

        self.root.after(33, self._update_frame)

    # ── FACE SCAN ───────────────────────────────────────
    def _scan_face(self, frame):
        self._set_status("● SCANNING", "#3ddc84")
        try:
            _, buf = cv2.imencode(".jpg", frame)
            person_id = self.face_matcher.match_face(buf.tobytes())

            if person_id:
                self.current_person  = person_id
                self.has_recognized  = True          # lock further scans
                self._set_person_badge(f"✦ {person_id.upper()}", "#c8a96e")
                self._set_bottom(f"Recognized: {person_id} — recording started automatically")
                self._set_status("● RECOGNIZED", "#c8a96e")
                self.root.after(0, lambda: self.record_btn.configure(
                    state="normal", bg="#e84545", text="⏹  STOP RECORDING"
                ))
                self._start_recording()
                self._load_summary(person_id)
            else:
                self.current_person = None
                self._set_person_badge("NO FACE DETECTED", "#444455")
                self._set_bottom("No recognized face in frame.")
                self.root.after(0, lambda: self.record_btn.configure(
                    state="disabled", bg="#555566", text="⏺  START RECORDING"
                ))
        except Exception as e:
            self._set_bottom(f"Scan error: {e}")

    # ── RECORDING ───────────────────────────────────────
    def _toggle_recording(self):
        if self.is_recording:
            self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self):
        if self.is_recording:
            return
        self.is_recording = True
        self.transcriber.start_recording()
        self.root.after(0, lambda: (
            self.record_btn.configure(bg="#e84545", text="⏹  STOP RECORDING"),
            self.rec_status.configure(text="🔴 Recording...")
        ))
        self._set_bottom("Recording conversation... press STOP when done.")

    def _stop_recording(self):
        if not self.is_recording:
            return
        self.is_recording   = False
        self.has_recognized = False      # allow next face scan after this session
        self.root.after(0, lambda: (
            self.record_btn.configure(bg="#555566", text="⏺  START RECORDING"),
            self.rec_status.configure(text="⏳ Transcribing...")
        ))
        self._set_bottom("Transcribing with Amazon Transcribe... please wait (~30 sec)")
        threading.Thread(target=self._process_transcript, daemon=True).start()

    def _process_transcript(self):
        try:
            transcript = self.transcriber.stop_and_transcribe()

            if transcript:
                # show transcript on screen
                self._set_transcript(transcript)

                # store as interaction in DynamoDB
                if self.current_person:
                    self.interaction_store.add_interaction(self.current_person, transcript)
                    self._set_bottom("Transcript saved! Generating new summary...")

                    # refresh summary with new interaction included
                    self._load_summary(self.current_person)
                    self.root.after(0, lambda: self.rec_status.configure(text="✅ Saved"))
            else:
                self._set_bottom("No speech detected. Try again.")
                self.root.after(0, lambda: self.rec_status.configure(text=""))

        except Exception as e:
            self._set_bottom(f"Transcribe error: {e}")
            self.root.after(0, lambda: self.rec_status.configure(text=""))

    # ── SUMMARY ─────────────────────────────────────────
    def _load_summary(self, person_id):
        try:
            interactions = self.interaction_store.get_interactions(person_id)
            self._update_log(interactions)
            if interactions:
                self._set_bottom("Generating summary with Amazon Bedrock...")
                summary = self.summary_gen.generate(person_id, interactions)
                self._set_summary(summary)
                self._set_bottom(f"Summary ready for {person_id}.")
            else:
                self._set_summary("No past interactions yet — start recording!")
        except Exception as e:
            self._set_bottom(f"Summary error: {e}")

    # ── UI HELPERS ───────────────────────────────────────
    def _set_status(self, text, color):
        self.root.after(0, lambda: self.status_dot.configure(text=text, fg=color))

    def _set_bottom(self, text):
        self.root.after(0, lambda: self.bottom_status.configure(text=text))

    def _set_person_badge(self, text, color):
        self.root.after(0, lambda: self.person_badge.configure(text=text, fg=color))

    def _set_transcript(self, text):
        def _do():
            self.transcript_text.configure(state="normal")
            self.transcript_text.delete("1.0", tk.END)
            self.transcript_text.insert(tk.END, text)
            self.transcript_text.configure(state="disabled")
        self.root.after(0, _do)

    def _set_summary(self, text):
        def _do():
            self.summary_text.configure(state="normal")
            self.summary_text.delete("1.0", tk.END)
            self.summary_text.insert(tk.END, text)
            self.summary_text.configure(state="disabled")
        self.root.after(0, _do)

    def _update_log(self, interactions):
        def _do():
            self.log_text.configure(state="normal")
            self.log_text.delete("1.0", tk.END)
            for r in interactions:
                ts = r.timestamp[:10]
                self.log_text.insert(tk.END, f"[{ts}]  {r.description}\n\n")
            self.log_text.configure(state="disabled")
        self.root.after(0, _do)

    def on_close(self):
        self.running = False
        if self.is_recording:
            self.transcriber.recording = False
        self.cap.release()
        self.root.destroy()


# ── ENTRY ────────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    app = MemoireApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
