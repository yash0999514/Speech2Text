import ttkbootstrap as tb
from ttkbootstrap.constants import *
from tkinter import scrolledtext, messagebox, filedialog
from PIL import Image, ImageTk
import speech_recognition as sr
import threading
import json, os

APP_TITLE = "Speech to Text App"
LOGO_PATH = "modern_mic.png"   # Use your modern mic icon here
CONFIG_FILE = "config.json"    # For saving theme settings

# ------------------- Config Handling -------------------
def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_config(data):
    with open(CONFIG_FILE, "w") as f:
        json.dump(data, f)

config = load_config()
default_theme = config.get("theme", "superhero")

# ------------------- Tooltips -------------------
def create_tooltip(widget, text):
    tooltip = tb.Toplevel(widget)
    tooltip.withdraw()
    tooltip.overrideredirect(True)
    label = tb.Label(tooltip, text=text, bootstyle="inverse-dark")
    label.pack(ipadx=5, ipady=3)
    def enter(event):
        x, y = event.x_root + 20, event.y_root + 10
        tooltip.geometry(f"+{x}+{y}")
        tooltip.deiconify()
    def leave(event):
        tooltip.withdraw()
    widget.bind("<Enter>", enter)
    widget.bind("<Leave>", leave)

# ------------------- Transcription -------------------
def transcribe_audio_file():
    file_path = filedialog.askopenfilename(
        title="Select Audio File", 
        filetypes=[("Audio Files", "*.wav *.mp3 *.flac")]
    )
    if not file_path:
        return
    recognizer = sr.Recognizer()
    try:
        if file_path.endswith('.mp3'):
            try:
                from pydub import AudioSegment
            except ImportError:
                messagebox.showerror("Dependency Missing", "Please install pydub: pip install pydub")
                return
            sound = AudioSegment.from_mp3(file_path)
            wav_path = file_path + ".wav"
            sound.export(wav_path, format="wav")
            file_path = wav_path
        with sr.AudioFile(file_path) as source:
            # Noise Optimizer based on selection
            mode = noise_mode_var.get()
            if mode == "Home":
                set_status("Applying home noise optimization...")
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
            elif mode == "Office":
                set_status("Applying office noise optimization...")
                recognizer.adjust_for_ambient_noise(source, duration=1.0)
            elif mode == "Other":
                set_status("Applying general noise optimization...")
                recognizer.adjust_for_ambient_noise(source, duration=0.8)

            set_status("Transcribing file...")
            audio = recognizer.record(source)
            text = recognizer.recognize_google(audio)
            text_area.insert(END, f"Transcribed Text:\n{text}\n\n")
            set_status("Idle")
    except Exception as e:
        set_status("Idle")
        messagebox.showerror("Error", str(e))

def live_subtitles():
    def recognize_loop():
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            # Noise Optimizer based on selection
            mode = noise_mode_var.get()
            if mode == "Home":
                set_status("Calibrating mic (home noise reduction)...")
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
            elif mode == "Office":
                set_status("Calibrating mic (office noise reduction)...")
                recognizer.adjust_for_ambient_noise(source, duration=1.0)
            elif mode == "Other":
                set_status("Calibrating mic (general noise reduction)...")
                recognizer.adjust_for_ambient_noise(source, duration=0.8)

            set_status("Listening...")
            while subtitle_running[0]:
                try:
                    audio = recognizer.listen(source, timeout=2, phrase_time_limit=3)
                    text = recognizer.recognize_google(audio)
                    text_area.insert(END, f"{text} ")
                    text_area.see(END)
                except sr.WaitTimeoutError:
                    continue
                except sr.UnknownValueError:
                    continue
                except sr.RequestError as e:
                    text_area.insert(END, f"\nAPI error: {e}\n")
                    break
        set_status("Idle")
    subtitle_running[0] = True
    threading.Thread(target=recognize_loop, daemon=True).start()

def stop_subtitles():
    subtitle_running[0] = False
    set_status("Idle")
    text_area.insert(END, "\nStopped live subtitles.\n")

def clear_text():
    text_area.delete('1.0', END)

# ------------------- Status Bar -------------------
def set_status(text):
    status_var.set(text)
    status_label.update_idletasks()

# ------------------- Theme Switch -------------------
def switch_theme(event=None):
    selected_theme = theme_combo.get()
    root.style.theme_use(selected_theme)
    save_config({"theme": selected_theme})  # save theme to config

# ------------------- Main Window -------------------
root = tb.Window(themename=default_theme)  # load saved theme
root.title(APP_TITLE)
root.geometry("950x750")
root.resizable(False, False)

# ------------------- Top Frame -------------------
top_frame = tb.Frame(root)
top_frame.pack(pady=(18, 0))

circle = tb.Canvas(top_frame, width=90, height=90, highlightthickness=0)
circle.create_oval(8, 8, 82, 82, fill="#ffffff", outline="#3b5998", width=5)
circle.pack(side=LEFT, padx=(10, 20))

try:
    logo_img = Image.open(LOGO_PATH).resize((60, 60), Image.ANTIALIAS)
    logo = ImageTk.PhotoImage(logo_img)
    circle.create_image(45, 45, image=logo)
except Exception:
    circle.create_text(45, 45, text="🎤", font=("Arial", 34))

app_label = tb.Label(top_frame, text="Speech to Text Converter", 
                     font=("Segoe UI", 28, "bold"))
app_label.pack(side=LEFT, pady=(7, 0))

# ------------------- Theme Switch Dropdown -------------------
theme_label = tb.Label(top_frame, text="Theme:", font=("Segoe UI", 12))
theme_label.pack(side=LEFT, padx=(40, 5))

theme_combo = tb.Combobox(
    top_frame, 
    values=sorted(root.style.theme_names()), 
    font=("Segoe UI", 11), 
    width=15, 
    state="readonly"
)
theme_combo.set(default_theme)
theme_combo.bind("<<ComboboxSelected>>", switch_theme)
theme_combo.pack(side=LEFT, padx=(0, 20))

# ------------------- Noise Optimizer Dropdown -------------------
noise_frame = tb.Frame(root)   # new row under top bar
noise_frame.pack(pady=(5, 0), anchor="w", padx=30)

noise_label = tb.Label(noise_frame, text="Noise Optimizer:", font=("Segoe UI", 12))
noise_label.pack(side=LEFT, padx=(0, 10))

noise_mode_var = tb.StringVar(value="Home")  # default Home
noise_combo = tb.Combobox(
    noise_frame,
    textvariable=noise_mode_var,
    values=["Home", "Office", "Other"],
    font=("Segoe UI", 11),
    width=15,
    state="readonly"
)
noise_combo.pack(side=LEFT)

# ------------------- Main Text Area -------------------
main_frame = tb.Frame(root, bootstyle="secondary")
main_frame.pack(pady=(10, 0), padx=24, fill="both", expand=False)

text_area = scrolledtext.ScrolledText(
    main_frame, width=104, height=16, wrap="word",
    font=("Segoe UI", 13), bg="#f5f5f5", fg="#1a1a1a", borderwidth=0
)
text_area.pack(padx=18, pady=14)

# ------------------- Buttons -------------------
button_frame = tb.Frame(root)
button_frame.pack(side=BOTTOM, fill=X, pady=(0,0), padx=24)

live_btn = tb.Button(button_frame, text="Start Live Subtitles", command=live_subtitles, bootstyle=SUCCESS)
live_btn.grid(row=0, column=0, sticky="ew", padx=6, pady=3)
create_tooltip(live_btn, "Transcribe your speech live as subtitles.")

stop_btn = tb.Button(button_frame, text="Stop Subtitles", command=stop_subtitles, bootstyle=WARNING)
stop_btn.grid(row=0, column=1, sticky="ew", padx=6, pady=3)
create_tooltip(stop_btn, "Stop listening and subtitle transcription.")

file_btn = tb.Button(button_frame, text="Transcribe Audio File", command=transcribe_audio_file, bootstyle=INFO)
file_btn.grid(row=0, column=2, sticky="ew", padx=6, pady=3)
create_tooltip(file_btn, "Convert an audio file (wav, mp3, flac) to text.")

clear_btn = tb.Button(button_frame, text="Clear", command=clear_text, bootstyle=SECONDARY)
clear_btn.grid(row=0, column=3, sticky="ew", padx=6, pady=3)
create_tooltip(clear_btn, "Clear all text.")

exit_btn = tb.Button(button_frame, text="❌ Exit", command=root.quit, bootstyle=DANGER)
exit_btn.grid(row=0, column=4, sticky="ew", padx=6, pady=3)
create_tooltip(exit_btn, "Exit the app.")

for i in range(5):
    button_frame.grid_columnconfigure(i, weight=1)

# ------------------- Status Bar -------------------
status_var = tb.StringVar(value="Idle")
status_label = tb.Label(root, textvariable=status_var, anchor=W, bootstyle="inverse-dark")
status_label.pack(fill=X, side=BOTTOM, ipady=4)

subtitle_running = [False]

root.mainloop()
