import tkinter as tk
from tkinter import font as tkfont
import threading
import time
import cv2
import numpy as np
from PIL import Image, ImageTk, ImageDraw
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

COL_NAME_BG    = (52, 168, 83, 220)
COL_TAG_BG     = (66, 133, 244, 220)
COL_CARD_BG    = (20, 20, 20, 190)
COL_WHITE      = (255, 255, 255, 255)
COL_LIGHT      = (220, 220, 220, 255)
COL_TRANSCRIPT = (180, 230, 180, 255)
COL_STATUS_OK  = (80, 200, 80, 200)
COL_STATUS_REC = (220, 60, 60, 200)


def rounded_rect(draw, xy, radius, fill):
    draw.rounded_rectangle(xy, radius=radius, fill=fill)


class MemoireApp:
    def __init__(self, root):
        self.root = root
        self.root.title("MemoireAR")
        self.root.configure(bg="#000000")
        self.root.geometry("900x620")
        self.root.resizable(False, False)

        self.face_matcher      = FaceMatcher(COLLECTION_ID, BUCKET_NAME, CONFIDENCE)
        self.interaction_store = InteractionStore(TABLE_NAME)
        self.summary_gen       = SummaryGenerator()
        self.transcriber       = TranscribeHandler()
        self.transcriber.on_interim_text = self._on_interim

        self.current_person     = None
        self.running            = True
        self.last_match_time    = 0
        self.cooldown           = 15
        self.is_recording       = False
        self.has_recognized     = False

        self.overlay_name       = ""
        self.overlay_tag        = ""
        self.overlay_summary    = ""
        self.overlay_transcript = ""
        self.overlay_status     = "SCANNING FOR FACES"
        self.overlay_recording  = False

        self._build_ui()
        self._start_camera()

    def _build_ui(self):
        lf = tkfont.Font(family="Courier New", size=10, weight="bold")
        sf = tkfont.Font(family="Courier New", size=9)

        self.canvas = tk.Canvas(self.root, width=900, height=560,
                                bg="#000000", highlightthickness=0)
        self.canvas.pack()

        bar = tk.Frame(self.root, bg="#0a0a0f", height=60)
        bar.pack(fill="x")
        bar.pack_propagate(False)

        self.status_lbl = tk.Label(bar, text="MEMOIRE  |  System ready",
                                   font=sf, fg="#555566", bg="#0a0a0f")
        self.status_lbl.pack(side="left", padx=16)

        tk.Button(bar, text="NEW PERSON", font=lf, fg="#ffffff", bg="#333344",
                  relief="flat", padx=12, pady=6,
                  command=self._reset_for_new_person).pack(side="right", padx=8, pady=8)

        self.rec_btn = tk.Button(bar, text="START RECORDING",
                                 font=lf, fg="#0a0a0f", bg="#555566",
                                 relief="flat", padx=12, pady=6,
                                 state="disabled",
                                 command=self._toggle_recording)
        self.rec_btn.pack(side="right", padx=4, pady=8)

        ef = tk.Frame(bar, bg="#0a0a0f")
        ef.pack(side="right", padx=8, pady=8)
        tk.Label(ef, text="TAG:", font=sf, fg="#555566", bg="#0a0a0f").pack(side="left")
        self.tag_entry = tk.Entry(ef, font=lf, fg="#e8e0d0", bg="#1a1a2e",
                                  insertbackground="#c8a96e", width=12, relief="flat")
        self.tag_entry.insert(0, "Son/Daughter")
        self.tag_entry.pack(side="left", padx=(4,0), ipady=4)

    def _start_camera(self):
        self.cap = cv2.VideoCapture(0)
        self._update_frame()

    def _update_frame(self):
        if not self.running:
            return
        ret, frame = self.cap.read()
        if ret:
            self.current_frame = frame
            display = self._draw_overlay(frame)
            img = Image.fromarray(cv2.cvtColor(display, cv2.COLOR_BGR2RGB))
            img = img.resize((900, 560))
            imgtk = ImageTk.PhotoImage(img)
            self.canvas.create_image(0, 0, anchor="nw", image=imgtk)
            self.canvas.image = imgtk

            if (time.time() - self.last_match_time > self.cooldown
                    and not self.is_recording
                    and not self.has_recognized):
                self.last_match_time = time.time()
                threading.Thread(target=self._scan_face,
                                 args=(frame,), daemon=True).start()

        self.root.after(33, self._update_frame)

    def _draw_overlay(self, frame):
        h, w = frame.shape[:2]
        pil  = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGBA")
        over = Image.new("RGBA", pil.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(over)

        # status pill top-left
        sc = COL_STATUS_REC if self.overlay_recording else COL_STATUS_OK
        st = ("[REC] " if self.overlay_recording else "[ ] ") + self.overlay_status
        sw = min(len(st) * 8 + 28, w - 12)
        rounded_rect(draw, [12, 12, sw, 36], 8, sc)
        draw.text((20, 15), st, fill=COL_WHITE)

        if self.overlay_name:
            x = w // 2 - 10
            y = h // 2 - 90

            # name badge
            nw = len(self.overlay_name) * 13 + 24
            rounded_rect(draw, [x, y, x + nw, y + 32], 10, COL_NAME_BG)
            draw.text((x + 10, y + 7), self.overlay_name.upper(), fill=COL_WHITE)

            # tag badge
            if self.overlay_tag:
                tx = x + nw + 8
                tw = len(self.overlay_tag) * 9 + 20
                rounded_rect(draw, [tx, y, tx + tw, y + 32], 10, COL_TAG_BG)
                draw.text((tx + 8, y + 7), self.overlay_tag, fill=COL_WHITE)

            # summary card
            if self.overlay_summary:
                lines  = self._wrap(self.overlay_summary, 44)[:4]
                card_h = len(lines) * 24 + 20
                rounded_rect(draw, [x, y + 40, w - 16, y + 40 + card_h],
                             12, COL_CARD_BG)
                for i, line in enumerate(lines):
                    draw.text((x + 12, y + 50 + i * 24), line, fill=COL_LIGHT)

        # transcript bottom
        if self.overlay_transcript:
            lines  = self._wrap(self.overlay_transcript, 70)[-2:]
            ch     = len(lines) * 22 + 14
            rounded_rect(draw, [10, h - ch - 10, w - 10, h - 10],
                         10, (0, 0, 0, 160))
            for i, line in enumerate(lines):
                draw.text((18, h - ch - 2 + i * 22), line, fill=COL_TRANSCRIPT)

        # red dot
        if self.overlay_recording:
            draw.ellipse([w - 28, 14, w - 12, 30], fill=(230, 50, 50, 230))

        merged = Image.alpha_composite(pil, over).convert("RGB")
        return cv2.cvtColor(np.array(merged), cv2.COLOR_RGB2BGR)

    def _wrap(self, text, width):
        words, lines, line = text.split(), [], ""
        for word in words:
            if len(line) + len(word) + 1 <= width:
                line += ("" if not line else " ") + word
            else:
                if line: lines.append(line)
                line = word
        if line: lines.append(line)
        return lines

    def _scan_face(self, frame):
        self.overlay_status = "SCANNING..."
        try:
            _, buf = cv2.imencode(".jpg", frame)
            person_id = self.face_matcher.match_face(buf.tobytes())
            if person_id:
                self.current_person = person_id
                self.has_recognized = True
                self.overlay_name   = person_id
                self.overlay_tag    = self.tag_entry.get().strip()
                self.overlay_status = "RECOGNIZED"
                self._set_status(f"Recognized: {person_id} — loading history...")
                self.root.after(0, lambda: self.rec_btn.configure(
                    state="normal", bg="#e84545", fg="#ffffff",
                    text="STOP RECORDING"))
                self._start_recording()
                # load existing summary immediately
                threading.Thread(target=self._load_summary,
                                 args=(person_id,), daemon=True).start()
            else:
                self.overlay_name    = ""
                self.overlay_summary = ""
                self.overlay_status  = "SCANNING FOR FACES"
                self._set_status("No recognized face.")
                self.root.after(0, lambda: self.rec_btn.configure(
                    state="disabled", bg="#555566", text="START RECORDING"))
        except Exception as e:
            self._set_status(f"Scan error: {e}")
            print(f"[Scan] {e}")

    def _toggle_recording(self):
        if self.is_recording: self._stop_recording()
        else: self._start_recording()

    def _start_recording(self):
        if self.is_recording: return
        self.is_recording      = True
        self.overlay_recording = True
        self.overlay_status    = "RECORDING"
        self.transcriber.start_recording()
        self.root.after(0, lambda: self.rec_btn.configure(
            bg="#e84545", fg="#ffffff", text="STOP RECORDING"))
        self._set_status("Recording... press STOP when done.")

    def _stop_recording(self):
        if not self.is_recording: return
        self.is_recording      = False
        self.overlay_recording = False
        self.overlay_status    = "TRANSCRIBING..."
        self.root.after(0, lambda: self.rec_btn.configure(
            bg="#aaaaaa", fg="#000000", text="TRANSCRIBING..."))
        self._set_status("Transcribing...")
        threading.Thread(target=self._process_transcript, daemon=True).start()

    def _process_transcript(self):
        try:
            transcript = self.transcriber.stop_and_transcribe()
            print(f"[App] Transcript: {transcript[:80] if transcript else 'EMPTY'}")
            if transcript:
                self.overlay_transcript = transcript
                if self.current_person:
                    self.interaction_store.add_interaction(self.current_person, transcript)
                    print(f"[App] Saved to DynamoDB for {self.current_person}")
                    self._set_status("Saved! Generating summary...")
                    self.overlay_status = "GENERATING SUMMARY..."
                    self._load_summary(self.current_person)
                    self.root.after(0, lambda: self.rec_btn.configure(
                        state="normal", bg="#3ddc84", fg="#000000",
                        text="RECORD MORE"))
            else:
                self.overlay_status = "RECOGNIZED"
                self._set_status("No speech detected. Try again.")
                self.root.after(0, lambda: self.rec_btn.configure(
                    state="normal", bg="#e84545", fg="#ffffff",
                    text="STOP RECORDING"))
        except Exception as e:
            print(f"[App] Transcript error: {e}")
            self._set_status(f"Error: {e}")

    def _on_interim(self, text):
        self.overlay_transcript = text

    def _load_summary(self, person_id):
        try:
            interactions = self.interaction_store.get_interactions(person_id)
            print(f"[App] {len(interactions)} interactions for {person_id}")
            if interactions:
                summary = self.summary_gen.generate(person_id, interactions)
                print(f"[App] Summary: {summary[:60]}")
                self.overlay_summary = summary
                self.overlay_status  = "RECOGNIZED"
                self._set_status(f"Summary ready for {person_id}")
            else:
                self.overlay_summary = "No past interactions yet. Start recording!"
                self.overlay_status  = "RECOGNIZED"
        except Exception as e:
            print(f"[App] Summary error: {e}")
            self.overlay_summary = "Summary unavailable."
            self.overlay_status  = "RECOGNIZED"

    def _reset_for_new_person(self):
        if self.is_recording: self._stop_recording()
        self.has_recognized     = False
        self.current_person     = None
        self.last_match_time    = 0
        self.overlay_name       = ""
        self.overlay_tag        = ""
        self.overlay_summary    = ""
        self.overlay_transcript = ""
        self.overlay_status     = "SCANNING FOR FACES"
        self.overlay_recording  = False
        self.root.after(0, lambda: self.rec_btn.configure(
            state="disabled", bg="#555566", fg="#0a0a0f",
            text="START RECORDING"))
        self._set_status("Ready for new person.")

    def _set_status(self, text):
        self.root.after(0, lambda: self.status_lbl.configure(
            text=f"MEMOIRE  |  {text}"))

    def on_close(self):
        self.running = False
        if self.is_recording: self.transcriber.recording = False
        self.cap.release()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = MemoireApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
