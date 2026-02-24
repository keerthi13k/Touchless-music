# air_piano.py
import cv2
import mediapipe as mp
import numpy as np
import simpleaudio as sa
import random
import time
from collections import deque

# =================== MediaPipe Setup ===================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Selfie segmentation (used only when reactive background is enabled)
mp_selfie = mp.solutions.selfie_segmentation
segmenter = mp_selfie.SelfieSegmentation(model_selection=1)

# =================== Audio Settings ===================
SAMPLE_RATE = 44100

def generate_tone(frequency, duration=0.4, instrument="piano"):
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
    if instrument == "piano":
        wave = (np.sin(frequency*2*np.pi*t) + 0.5*np.sin(2*frequency*2*np.pi*t)) * np.exp(-3*t)
    elif instrument == "guitar":
        saw = 2*(t*frequency - np.floor(0.5 + t*frequency))
        wave = saw * np.exp(-4*t)
    elif instrument == "flute":
        vibrato = 5*np.sin(2*np.pi*5*t)
        wave = np.sin(2*np.pi*(frequency+vibrato)*t) * 0.8
    elif instrument == "synth":
        wave = np.sin(2*np.pi*frequency*t) + 0.5*np.sin(2*np.pi*(frequency*1.01)*t)
    else:
        wave = np.sin(2*np.pi*frequency*t)
    wave *= 0.3
    audio = (wave * 32767).astype(np.int16)
    return audio

# =================== Notes & Instruments ===================
NOTES = {"C":261.63, "D":293.66, "E":329.63, "F":349.23, "G":392.00, "A":440.00, "B":493.88}
keys = list(NOTES.keys())

instruments = ["piano", "guitar", "flute", "synth"]
current_instrument_index = 0
current_instrument = instruments[current_instrument_index]

# =================== Particles ===================
particles = []

def add_particles(x, y, color):
    for _ in range(25):
        particles.append({
            "x": x, "y": y,
            "vx": random.uniform(-2, 2),
            "vy": random.uniform(-4, -1),
            "color": color,
            "lifetime": random.randint(20, 40)
        })

def update_particles(frame):
    for p in particles[:]:
        p["x"] += p["vx"]
        p["y"] += p["vy"]
        p["lifetime"] -= 1
        if p["lifetime"] <= 0:
            particles.remove(p)
        else:
            cv2.circle(frame, (int(p["x"]), int(p["y"])), 5, p["color"], -1)
            cv2.circle(frame, (int(p["x"]), int(p["y"])), 15,
                       (p["color"][0], p["color"][1], p["color"][2], 50), 1)

# =================== Song Mode ===================
songs = {
    "Twinkle": {
        "notes": ["C", "C", "G", "G", "A", "A", "G"],
        "lyrics": ["Twin", "kle", "Twin", "kle", "Lit", "tle", "Star"]
    },
    "HappyBirthday": {
        "notes": ["C", "C", "D", "C", "F", "E"],
        "lyrics": ["Happy", "Birth", "day", "to", "You", "!"]
    }
}
song_mode = False
current_song = None
song_index = 0

# =================== Fingertip Trail ===================
trail_length = 15
fingertip_trail = deque(maxlen=trail_length)

# =================== Reactive Background Parameters ===================
bg_enabled = False           # Toggle with 'b' key
bg_base_hue = 90
bg_target_hue = bg_base_hue
bg_hue = float(bg_base_hue)
bg_pulse = 0.0
PULSE_DECAY = 0.90
HUE_SMOOTH = 0.15

key_hues = {"C":15, "D":30, "E":45, "F":75, "G":95, "A":125, "B":155}

def make_gradient_bg(h, w, base_hue, pulse):
    # Smooth gradient HSV background; pulse affects brightness/saturation
    v_top = int(140 + 25 * pulse)
    v_bottom = int(200 + 40 * pulse)
    v_col = np.linspace(v_top, v_bottom, h, dtype=np.uint8)
    V = np.repeat(v_col[:, None], w, axis=1)
    s_val = int(170 + 60 * pulse)
    S = np.full((h, w), np.clip(s_val, 0, 255), dtype=np.uint8)
    hue_shift = int(10 * np.sin(time.time()*0.5))
    hue_left = (base_hue + hue_shift - 8) % 180
    hue_right = (base_hue + hue_shift + 8) % 180
    H_row = np.linspace(hue_left, hue_right, w, dtype=np.float32)
    H = np.repeat(H_row[None, :], h, axis=0).astype(np.uint8)
    hsv = np.dstack([H, S, V])
    bg = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bg

# =================== Camera ===================
cap = cv2.VideoCapture(0)

last_note = None  # for sound-on-change logic

# Helper to highlight expected key during Song Mode
def expected_key_for_song():
    if not current_song:
        return None
    notes = songs[current_song]["notes"]
    if song_index < len(notes):
        return notes[song_index]
    return None

# =================== Main Loop ===================
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    keyboard_top = int(h * 0.65)
    keyboard_bottom = h

    # Compose scene: reactive background if enabled, otherwise raw camera frame
    if bg_enabled:
        rgb_for_seg = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        seg_res = segmenter.process(rgb_for_seg)
        mask = seg_res.segmentation_mask
        fg_mask = (mask > 0.5).astype(np.uint8)
        # Smooth hue toward target and build background canvas
        bg_hue = (1 - HUE_SMOOTH) * bg_hue + HUE_SMOOTH * bg_target_hue
        bg_canvas = make_gradient_bg(h, w, int(bg_hue) % 180, np.clip(bg_pulse, 0.0, 1.0))
        fg_mask_3 = np.dstack([fg_mask]*3)
        scene = frame * fg_mask_3 + bg_canvas * (1 - fg_mask_3)
        scene = scene.astype(np.uint8)
    else:
        scene = frame.copy()

    # Run hand detection on the composed scene (so overlays appear above BG)
    rgb = cv2.cvtColor(scene, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    fingertip_x, fingertip_y = None, None
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            fingertip_x = int(hand_landmarks.landmark[8].x * w)
            fingertip_y = int(hand_landmarks.landmark[8].y * h)
            mp_draw.draw_landmarks(scene, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            fingertip_trail.append((fingertip_x, fingertip_y))

    # Draw keys & detect current pressed key only (green when finger is physically over it)
    key_width = w // len(keys)
    pressed_note = None
    expected_note = expected_key_for_song()

    for i, key in enumerate(keys):
        x1 = i * key_width
        x2 = x1 + key_width
        color = (255, 255, 255)  # default white

        # If Song Mode active and this is the expected key, highlight it yellow (visual guidance)
        if song_mode and current_song and expected_note == key:
            # but don't override the green when finger is actually pressing
            highlight_color = (0, 255, 255)  # yellow-ish (BGR)
        else:
            highlight_color = None

        # If fingertip is currently over this key -> current key highlight (green)
        if fingertip_x and x1 < fingertip_x < x2 and fingertip_y is not None and fingertip_y > keyboard_top:
            color = (0, 255, 0)
            pressed_note = key
        elif highlight_color is not None:
            color = highlight_color

        # Draw key
        cv2.rectangle(scene, (x1, keyboard_top), (x2, keyboard_bottom), color, -1)
        cv2.rectangle(scene, (x1, keyboard_top), (x2, keyboard_bottom), (0, 0, 0), 2)
        cv2.putText(scene, key, (x1 + 20, keyboard_bottom - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # --- Sound + effects logic ---
    # If user pressed the expected key in Song Mode, advance; otherwise accept any live press (plays sound)
    if pressed_note is not None:
        # play sound only when note changes (prevents continuous retrigger)
        if pressed_note != last_note:
            freq = NOTES[pressed_note]
            audio = generate_tone(freq, instrument=current_instrument)
            sa.play_buffer(audio, 1, 2, SAMPLE_RATE)

            # particles
            idx = keys.index(pressed_note)
            x_center = int((idx + 0.5) * key_width)
            add_particles(x_center, keyboard_top - 20, (random.randint(50,255), random.randint(50,255), random.randint(50,255)))

            # If Song Mode: only advance when the pressed note matches expected
            if song_mode and current_song:
                if song_index < len(songs[current_song]["notes"]) and pressed_note == songs[current_song]["notes"][song_index]:
                    song_index += 1
                    # background reacts to song advancement
                    bg_target_hue = key_hues.get(pressed_note, bg_base_hue)
                    bg_pulse = 1.0
                else:
                    # If wrong key in song mode, do not advance; but still small visual feedback was already shown
                    pass
            else:
                # when playing live (not in guided song), background reactive on presses too
                bg_target_hue = key_hues.get(pressed_note, bg_base_hue)
                bg_pulse = max(bg_pulse, 0.6)

            last_note = pressed_note
    else:
        # finger not on any key — allow re-trigger on next touch
        last_note = None

    # Decay pulse for background smoothing
    bg_pulse *= PULSE_DECAY

    # Draw & update particles on scene
    update_particles(scene)

    # Fingertip trail (magic wand)
    for idx, point in enumerate(fingertip_trail):
        alpha = (idx + 1) / trail_length
        overlay = scene.copy()
        cv2.circle(overlay, point, 15, (0, 255, 255), -1)
        cv2.addWeighted(overlay, alpha * 0.6, scene, 1 - alpha * 0.6, 0, scene)

    # Song Mode lyrics display and finished message
    if song_mode and current_song:
        lyrics = songs[current_song]["lyrics"]
        display_text = ""
        for idx, word in enumerate(lyrics):
            if idx == song_index:
                display_text += f"[{word}] "
            else:
                display_text += f"{word} "
        cv2.putText(scene, display_text.strip(), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
        if song_index >= len(songs[current_song]["notes"]):
            cv2.putText(scene, f"🎉 Finished {current_song}!", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    # HUD & instructions
    cv2.putText(scene, f"Instrument: {current_instrument}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
    mode_text = "Reactive BG: ON" if bg_enabled else "Reactive BG: OFF"
    cv2.putText(scene, f"{mode_text}  (press 'b' to toggle)", (10, int(h*0.95)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(scene, "Press 'i'=switch instr | 's'=toggle song | 1/2=select song | ESC=quit", (10, int(h*0.90)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    cv2.imshow("Air Piano Expo", scene)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break
    elif key == ord('i'):
        current_instrument_index = (current_instrument_index + 1) % len(instruments)
        current_instrument = instruments[current_instrument_index]
        bg_target_hue = (bg_target_hue + 20) % 180
        bg_pulse = max(bg_pulse, 0.5)
    elif key == ord('s'):
        # toggle Song Mode; keep current song selection if already chosen
        song_mode = not song_mode
        song_index = 0
        last_note = None
        if song_mode and current_song:
            # when enabling guided mode and a song is loaded, visually pulse the bg to indicate start
            bg_target_hue = key_hues.get(expected_key_for_song() or keys[0], bg_base_hue)
            bg_pulse = 0.9
    elif key == ord('1'):
        if song_mode:
            current_song = "Twinkle"
            song_index = 0
            last_note = None
            # pulse when song loaded
            bg_target_hue = key_hues.get(expected_key_for_song() or keys[0], bg_base_hue)
            bg_pulse = 0.9
    elif key == ord('2'):
        if song_mode:
            current_song = "HappyBirthday"
            song_index = 0
            last_note = None
            bg_target_hue = key_hues.get(expected_key_for_song() or keys[0], bg_base_hue)
            bg_pulse = 0.9
    elif key == ord('b'):
        # toggle reactive background
        bg_enabled = not bg_enabled
        bg_pulse = 0.8 if bg_enabled else 0.0
        bg_hue = float(bg_target_hue)

cap.release()
cv2.destroyAllWindows()
