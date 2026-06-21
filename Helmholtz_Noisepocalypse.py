"""Helmholtz and the Noisepocalypse: EOG filter game.

The program combines a live filter builder, guided EOG calibration, a
controlled left/right test, a three-lane runner, CSV logging, and
time/frequency-domain analysis.
"""

import csv
import glob
import queue
import random
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pygame
import serial
from scipy.signal import welch

# --------------------------- Settings ---------------------------

PREFERRED_PORT = "/dev/cu.usbmodem101" # # port prüfen mit: ls /dev/cu.* (at least for Mac)
BAUD_RATE = 115200
ADC_MINIMUM = 0
ADC_MAXIMUM = 65535

WIDTH = 1000
HEIGHT = 700
FPS = 60
BACKGROUND = (18, 20, 28)
FOREGROUND = (235, 235, 240)
MUTED = (160, 165, 180)
ACCENT = (80, 190, 255)
DANGER = (240, 90, 90)
SUCCESS = (90, 220, 140)

ROAD_LEFT = 170
ROAD_RIGHT = WIDTH - 170
ROAD_TOP = 90
ROAD_BOTTOM = HEIGHT - 40
LANE_COUNT = 3
LANE_WIDTH = (ROAD_RIGHT - ROAD_LEFT) / LANE_COUNT
LANE_CENTERS = [
    ROAD_LEFT + LANE_WIDTH * (index + 0.5)
    for index in range(LANE_COUNT)
]

GAME_DURATION_SECONDS = 60.0
STARTING_LIVES = 5
PLAYER_WIDTH = 74
PLAYER_HEIGHT = 46
PLAYER_Y = HEIGHT - 120
BASE_OBSTACLE_SPEED = 250.0
SPEED_INCREASE_PER_SECOND = 1.2
ROUND_RANDOM_SEED = 42
POINTS_PER_AVOIDED_OBSTACLE = 100
COLLISION_PENALTY = 100

# A "gate row" is one descending two-/three-alternative forced choice: exactly
# one lane holds the clean signal gate, the others hold noise artifacts. The
# player must steer the impulse into the clean gate, so every row demands one
# clear LEFT/RIGHT decision.
GATE_HEIGHT = 58
GATE_SPAWN_INTERVAL = 1.25
# Quick tutorial window at the start of the runner that labels the impulse and
# explains the controls before the first gate appears.
ONBOARDING_SECONDS = 3.0

CALIBRATION_CENTER_SECONDS = 3.0
CALIBRATION_CUE_SECONDS = 1.3
CALIBRATION_RETURN_SECONDS = 1.5
CALIBRATION_REPETITIONS = 5

# Ignore the first part of each center phase because it contains the
# return eye movement and filter settling.
CALIBRATION_CENTER_SETTLE_SECONDS = 0.45

# Use the main part of a cue phase, not the exact screen transition.
CALIBRATION_RESPONSE_START_SECONDS = 0.10
CALIBRATION_RESPONSE_END_SECONDS = 1.05

MINIMUM_CALIBRATION_RESPONSE = 50.0

# Controlled scientific test before the free runner.
SCIENCE_NUMBER_OF_TRIALS = 12
SCIENCE_FIXATION_SECONDS = 1.00
SCIENCE_RESPONSE_SECONDS = 1.40
SCIENCE_RETURN_SECONDS = 0.80
SCIENCE_RANDOM_SEED = 2026

# Number of recent samples shown in the live filter preview.
PREVIEW_BUFFER_SAMPLES = 450

NOMINAL_SAMPLING_RATE_HZ = 100.0
PSD_SEGMENT_DURATION_SECONDS = 4.0
PSD_MAX_FREQUENCY_HZ = 20.0
POWER_EPSILON = 1e-20

DATA_DIRECTORY = Path("data")
PLOT_DIRECTORY = Path("plots")
RESULT_DIRECTORY = Path("results")
for folder in (DATA_DIRECTORY, PLOT_DIRECTORY, RESULT_DIRECTORY):
    folder.mkdir(exist_ok=True)

# Put the Helmholtz image here (any location works):
#   images/helmholtz_help.png
#   assets/helmholtz_help.png
#   helmholtz_help.png
# Paths are resolved relative to this Python file, so the game also works
# when it is started from another working directory.
BASE_DIRECTORY = Path(__file__).resolve().parent
HELMHOLTZ_IMAGE_CANDIDATES = [
    BASE_DIRECTORY / "images" / "helmholtz_help.png",
    BASE_DIRECTORY / "assets" / "helmholtz_help.png",
    BASE_DIRECTORY / "helmholtz_help.png",
]
HELMHOLTZ_IMAGE_PATH = next(
    (path for path in HELMHOLTZ_IMAGE_CANDIDATES if path.exists()),
    HELMHOLTZ_IMAGE_CANDIDATES[0],
)

FILTER_TYPES = ["Raw", "Highpass", "EMA Smooth", "Moving Average"]


# --------------------------- Filters ---------------------------

@dataclass
class FilterConfig:
    filter_type: str = "EMA Smooth"
    baseline_alpha: float = 0.995
    smoothing_alpha: float = 0.80
    moving_average_window: int = 8
    threshold_multiplier: float = 1.00
    refractory_ms: int = 350


class FilterProcessor:
    def __init__(self, config):
        self.config = config
        self.reset()

    def reset(self):
        self.baseline = None
        self.smooth = 0.0
        self.moving_values = deque(maxlen=self.config.moving_average_window)

    def process(self, raw_value):
        raw_value = float(raw_value)
        if self.baseline is None:
            self.baseline = raw_value

        alpha = self.config.baseline_alpha
        self.baseline = alpha * self.baseline + (1.0 - alpha) * raw_value
        highpass = raw_value - self.baseline

        smooth_alpha = self.config.smoothing_alpha
        self.smooth = smooth_alpha * self.smooth + (1.0 - smooth_alpha) * highpass

        self.moving_values.append(highpass)
        moving_average = float(np.mean(self.moving_values))

        if self.config.filter_type == "Raw":
            filtered = raw_value
        elif self.config.filter_type == "Highpass":
            filtered = highpass
        elif self.config.filter_type == "EMA Smooth":
            filtered = self.smooth
        elif self.config.filter_type == "Moving Average":
            filtered = moving_average
        else:
            raise ValueError(f"Unknown filter type: {self.config.filter_type}")

        return self.baseline, highpass, filtered


@dataclass
class CalibrationResult:
    center_level: float
    center_noise: float
    left_response: float
    right_response: float
    left_polarity: int
    right_polarity: int
    left_threshold: float
    right_threshold: float
    neutral_threshold: float
    valid: bool
    warning: str = ""


class EyeMovementDetector:
    def __init__(self, calibration, refractory_ms):
        self.calibration = calibration
        self.refractory_seconds = refractory_ms / 1000.0
        self.state = "ready"
        self.last_event_time = -1e9
        self.lock_start_time = -1e9

    def reset(self):
        """Arm the detector for a new controlled trial or game round."""

        self.state = "ready"
        self.last_event_time = -1e9
        self.lock_start_time = -1e9

    def update(self, filtered_value, timestamp):
        centered = float(filtered_value) - self.calibration.center_level

        if self.state == "locked":
            returned = abs(centered) <= self.calibration.neutral_threshold
            refractory_done = timestamp - self.last_event_time >= self.refractory_seconds
            lock_too_long = timestamp - self.lock_start_time >= 1.2
            if (returned and refractory_done) or lock_too_long:
                self.state = "ready"
            return None, centered

        if timestamp - self.last_event_time < self.refractory_seconds:
            return None, centered

        left_strength = self.calibration.left_polarity * centered
        right_strength = self.calibration.right_polarity * centered
        left_crossed = left_strength >= self.calibration.left_threshold
        right_crossed = right_strength >= self.calibration.right_threshold

        event = None
        if left_crossed and right_crossed:
            left_ratio = left_strength / max(self.calibration.left_threshold, 1e-9)
            right_ratio = right_strength / max(self.calibration.right_threshold, 1e-9)
            event = "left" if left_ratio > right_ratio else "right"
        elif left_crossed:
            event = "left"
        elif right_crossed:
            event = "right"

        if event is not None:
            self.state = "locked"
            self.last_event_time = timestamp
            self.lock_start_time = timestamp

        return event, centered


# --------------------------- Serial ---------------------------

def find_serial_port():
    if Path(PREFERRED_PORT).exists():
        return PREFERRED_PORT

    candidates = sorted(glob.glob("/dev/cu.usbmodem*"))
    if len(candidates) == 1:
        print(f"Automatically selected port: {candidates[0]}")
        return candidates[0]
    if not candidates:
        raise FileNotFoundError("No /dev/cu.usbmodem* port was found.")
    raise RuntimeError(
        "More than one serial port was found. Set PREFERRED_PORT:\n"
        + "\n".join(candidates)
    )


def parse_adc_value(raw_line):
    text = raw_line.decode(errors="ignore").strip()
    if not text:
        return None
    try:
        value = int(text)
    except ValueError:
        return None
    if not ADC_MINIMUM <= value <= ADC_MAXIMUM:
        return None
    return value


class SerialReader:
    def __init__(self, serial_connection):
        self.serial_connection = serial_connection
        self.samples = queue.Queue(maxsize=10000)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.error = None
        self.invalid_line_count = 0

    def start(self):
        self.thread.start()

    def _read_loop(self):
        while not self.stop_event.is_set():
            try:
                raw_line = self.serial_connection.readline()
            except serial.SerialException as error:
                self.error = error
                break

            value = parse_adc_value(raw_line)
            if value is None:
                if raw_line:
                    self.invalid_line_count += 1
                continue

            sample = (time.perf_counter(), value)
            try:
                self.samples.put_nowait(sample)
            except queue.Full:
                try:
                    self.samples.get_nowait()
                except queue.Empty:
                    pass
                self.samples.put_nowait(sample)

    def drain(self, maximum=1000):
        values = []
        for _ in range(maximum):
            try:
                values.append(self.samples.get_nowait())
            except queue.Empty:
                break
        return values

    def clear(self):
        self.drain(maximum=10000)

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=1.0)


def test_serial_stream(serial_reader, required_samples=10, timeout=5.0):
    print("Testing the ESP32 data stream...")
    deadline = time.perf_counter() + timeout
    received = 0

    while time.perf_counter() < deadline:
        if serial_reader.error is not None:
            print(serial_reader.error)
            return False
        received += len(serial_reader.drain())
        if received >= required_samples:
            print(f"Serial test successful: {received} valid samples received.")
            return True
        time.sleep(0.05)

    print("Serial test failed: not enough valid samples.")
    return False


# --------------------------- Drawing ---------------------------

def draw_text(screen, text, position, size=28, color=FOREGROUND, center=False, bold=False):
    font = pygame.font.SysFont(None, size, bold=bold)
    surface = font.render(str(text), True, color)
    rect = surface.get_rect()
    if center:
        rect.center = position
    else:
        rect.topleft = position
    screen.blit(surface, rect)


def wrap_text(text, font, maximum_width):
    """Split text into lines that fit inside maximum_width."""

    wrapped_lines = []

    for paragraph in str(text).split("\n"):
        if paragraph == "":
            wrapped_lines.append("")
            continue

        words = paragraph.split()
        current_line = words[0] if words else ""

        for word in words[1:]:
            candidate = f"{current_line} {word}"
            if font.size(candidate)[0] <= maximum_width:
                current_line = candidate
            else:
                wrapped_lines.append(current_line)
                current_line = word

        wrapped_lines.append(current_line)

    return wrapped_lines


def draw_wrapped_text(
    screen,
    text,
    rectangle,
    size=22,
    color=FOREGROUND,
    bold=False,
    line_spacing=4,
):
    """Draw wrapped text and return the y-coordinate below the last line."""

    font = pygame.font.SysFont(None, size, bold=bold)
    line_height = font.get_linesize() + line_spacing
    y = rectangle.top

    for line in wrap_text(text, font, rectangle.width):
        if line:
            surface = font.render(line, True, color)
            screen.blit(surface, (rectangle.left, y))
        y += line_height

    return y


def draw_wrapped_text_fit(
    screen,
    text,
    rectangle,
    maximum_size=22,
    minimum_size=14,
    color=FOREGROUND,
    bold=False,
    line_spacing=2,
):
    """Choose the largest font size that keeps all text inside a rectangle."""

    chosen_size = minimum_size
    chosen_lines = []
    chosen_line_height = 0

    for size in range(maximum_size, minimum_size - 1, -1):
        font = pygame.font.SysFont(None, size, bold=bold)
        lines = wrap_text(text, font, rectangle.width)
        line_height = font.get_linesize() + line_spacing

        if len(lines) * line_height <= rectangle.height:
            chosen_size = size
            chosen_lines = lines
            chosen_line_height = line_height
            break
    else:
        font = pygame.font.SysFont(None, minimum_size, bold=bold)
        chosen_lines = wrap_text(text, font, rectangle.width)
        chosen_line_height = font.get_linesize() + line_spacing

    font = pygame.font.SysFont(None, chosen_size, bold=bold)
    y = rectangle.top

    for line in chosen_lines:
        if line:
            surface = font.render(line, True, color)
            screen.blit(surface, (rectangle.left, y))
        y += chosen_line_height

    return y, chosen_size


def load_helmholtz_image(maximum_size):
    """Load and scale the optional Helmholtz image without crashing."""

    if not HELMHOLTZ_IMAGE_PATH.exists():
        return None

    try:
        image = pygame.image.load(str(HELMHOLTZ_IMAGE_PATH)).convert_alpha()
    except (pygame.error, OSError) as error:
        print(f"Could not load Helmholtz image: {error}")
        return None

    maximum_width, maximum_height = maximum_size
    scale = min(
        maximum_width / image.get_width(),
        maximum_height / image.get_height(),
    )
    new_size = (
        max(1, int(image.get_width() * scale)),
        max(1, int(image.get_height() * scale)),
    )
    return pygame.transform.smoothscale(image, new_size)


def draw_helmholtz_placeholder(screen, rectangle):
    """Draw a simple professor placeholder when no image file is present."""

    pygame.draw.rect(screen, (35, 39, 52), rectangle, border_radius=16)
    pygame.draw.rect(screen, (75, 82, 101), rectangle, width=2, border_radius=16)

    center_x = rectangle.centerx
    head_y = rectangle.top + 105

    # Hair, head, glasses and laboratory coat.
    pygame.draw.circle(screen, (215, 218, 225), (center_x, head_y), 58)
    pygame.draw.arc(
        screen,
        (105, 110, 125),
        pygame.Rect(center_x - 65, head_y - 68, 130, 70),
        3.1,
        6.2,
        width=10,
    )
    pygame.draw.circle(screen, BACKGROUND, (center_x - 23, head_y - 2), 17, width=3)
    pygame.draw.circle(screen, BACKGROUND, (center_x + 23, head_y - 2), 17, width=3)
    pygame.draw.line(screen, BACKGROUND, (center_x - 6, head_y - 2), (center_x + 6, head_y - 2), width=3)
    pygame.draw.arc(
        screen,
        BACKGROUND,
        pygame.Rect(center_x - 20, head_y + 12, 40, 25),
        0.1,
        3.0,
        width=3,
    )

    coat = [
        (rectangle.left + 52, rectangle.bottom - 20),
        (center_x - 48, head_y + 48),
        (center_x, head_y + 72),
        (center_x + 48, head_y + 48),
        (rectangle.right - 52, rectangle.bottom - 20),
    ]
    pygame.draw.polygon(screen, (225, 228, 235), coat)
    pygame.draw.line(screen, ACCENT, (center_x, head_y + 72), (center_x, rectangle.bottom - 24), width=4)

    draw_text(
        screen,
        "HELMHOLTZ",
        (rectangle.centerx, rectangle.bottom - 22),
        18,
        MUTED,
        center=True,
        bold=True,
    )


# --------------- Sci-fi control-panel style (story screen) ---------------

STORY_BG = (7, 11, 19)
STORY_PANEL = (12, 18, 30)
STORY_PANEL_DEEP = (9, 14, 23)
STORY_BORDER = (38, 84, 112)
STORY_CYAN = (95, 205, 240)
STORY_CYAN_DIM = (70, 140, 175)
STORY_TEXT = (198, 210, 226)
STORY_TEXT_DIM = (132, 148, 170)
STORY_TITLE = (205, 218, 234)


def draw_corner_brackets(screen, rectangle, color, length=16, width=2):
    """Draw L-shaped accents in the four corners of a rectangle."""

    left, top, right, bottom = (
        rectangle.left, rectangle.top, rectangle.right, rectangle.bottom,
    )
    for corner_x, corner_y, dx, dy in (
        (left, top, 1, 1),
        (right, top, -1, 1),
        (left, bottom, 1, -1),
        (right, bottom, -1, -1),
    ):
        pygame.draw.line(screen, color, (corner_x, corner_y), (corner_x + dx * length, corner_y), width)
        pygame.draw.line(screen, color, (corner_x, corner_y), (corner_x, corner_y + dy * length), width)


def draw_tech_panel(screen, rectangle, fill=STORY_PANEL, border=STORY_BORDER, bracket=STORY_CYAN):
    """Draw a dark control-panel box with a thin border and corner brackets."""

    pygame.draw.rect(screen, fill, rectangle, border_radius=6)
    pygame.draw.rect(screen, border, rectangle, width=1, border_radius=6)
    draw_corner_brackets(screen, rectangle, bracket)


def draw_mission_step(screen, x, y, number, title, description, width, height):
    """Draw one numbered mission step with a circular badge."""

    pygame.draw.circle(screen, STORY_CYAN, (x + 18, y + 18), 18, width=2)
    draw_text(screen, str(number), (x + 18, y + 19), 26, STORY_CYAN, center=True, bold=True)
    draw_text(screen, title, (x + 50, y + 5), 24, STORY_TEXT, bold=True)
    draw_wrapped_text_fit(
        screen,
        description,
        pygame.Rect(x + 50, y + 33, width - 50, height - 33),
        maximum_size=19,
        minimum_size=14,
        color=STORY_TEXT_DIM,
        line_spacing=1,
    )


def story_screen(screen, clock):
    """Show the game story once before the player enters the filter lab."""

    story_text = (
        "Oh no! After working for hours on his newest electrical signal machine, "
        "Professor Hermann von Helmholtz accidentally spilled an entire cup of "
        "coffee over the control panel.\n\n"
        "The machine had never tasted coffee before and it reacted by going "
        "completely out of control. The coffee short-circuited the system and "
        "sent every amplifier into overdrive. Dials began spinning, warning lights "
        "flashed, and thousands of chaotic electrical signals flooded the "
        "laboratory. With one enormous crackle, the machine unleashed the "
        "Noisepocalypse: a violent storm of baseline drift, muscle artifacts, "
        "electrode pops, and uncontrollable electrical noise.\n\n"
        "Now only one clean nerve impulse remains. To save it, you must connect "
        "yourself to Helmholtz's machine and guide the impulse through its three "
        "signal lanes using only your eyes."
    )

    panic_headline = "\"PLEASE HELP ME!"
    panic_sub = "You are my last hope!"
    panic_body = (
        "Your shield against noise and your only weapon against the "
        "Noisepocalypse is your... draaammaaa... FILTER!\n\n"
        "Build it carefully! A weak filter may react to every tiny disturbance, "
        "but a filter that is too strong may become too slow. Control the impulse, "
        "avoid the artifacts, and reach the Emergency Shutdown Core before the "
        "entire laboratory is swallowed by noise."
    )

    mission_steps = [
        ("CONNECT", "Link yourself to Helmholtz's machine."),
        ("GUIDE", "Steer the impulse using only your eyes."),
        ("SURVIVE", "Reach the Shutdown Core before the noise wins."),
    ]

    helmholtz_image = load_helmholtz_image((200, 215))

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return None
                if event.key in (pygame.K_SPACE, pygame.K_RETURN):
                    return "start"

        screen.fill(STORY_BG)
        pygame.draw.rect(
            screen, STORY_BORDER, pygame.Rect(14, 12, WIDTH - 28, HEIGHT - 24),
            width=1, border_radius=10,
        )

        # --- Title block (top left) ---
        draw_text(screen, "HELMHOLTZ", (40, 22), 62, STORY_TITLE, bold=True)
        draw_text(screen, "AND THE", (46, 86), 22, STORY_CYAN_DIM, bold=True)
        draw_text(screen, "NOISEPOCALYPSE", (40, 106), 50, STORY_TITLE, bold=True)
        draw_text(screen, "- THE GAME STORY -", (46, 156), 22, STORY_CYAN, bold=True)

        # --- Narration (wide, across the top) ---
        draw_wrapped_text_fit(
            screen,
            story_text,
            pygame.Rect(36, 182, WIDTH - 72, 176),
            maximum_size=23,
            minimum_size=15,
            color=STORY_TEXT,
            line_spacing=2,
        )

        # --- Helmholtz panic (bottom left) ---
        draw_text(screen, "HELMHOLTZ TURNS TO YOU IN PANIC:", (40, 364), 20, STORY_CYAN, bold=True)
        panic = pygame.Rect(36, 388, 640, 250)
        draw_tech_panel(screen, panic)

        portrait = pygame.Rect(panic.left + 16, panic.top + 16, 200, panic.height - 32)
        if helmholtz_image is None:
            draw_helmholtz_placeholder(screen, portrait)
        else:
            pygame.draw.rect(screen, STORY_PANEL_DEEP, portrait, border_radius=8)
            image_rect = helmholtz_image.get_rect(center=portrait.center)
            screen.blit(helmholtz_image, image_rect)

        speech_x = portrait.right + 24
        speech_width = panic.right - speech_x - 20
        draw_text(screen, panic_headline, (speech_x, panic.top + 16), 30, STORY_CYAN, bold=True)
        draw_text(screen, panic_sub, (speech_x, panic.top + 54), 22, STORY_TEXT, bold=True)
        draw_wrapped_text_fit(
            screen,
            panic_body,
            pygame.Rect(speech_x, panic.top + 92, speech_width, panic.height - 108),
            maximum_size=22,
            minimum_size=15,
            color=STORY_TEXT,
            line_spacing=2,
        )

        # --- Your mission (bottom right) ---
        mission = pygame.Rect(692, 388, 272, 250)
        draw_tech_panel(screen, mission)
        draw_text(screen, "YOUR MISSION", (mission.left + 20, mission.top + 16), 24, STORY_CYAN, bold=True)
        for index, (title, description) in enumerate(mission_steps):
            draw_mission_step(
                screen,
                mission.left + 20,
                mission.top + 58 + index * 64,
                index + 1,
                title,
                description,
                mission.width - 40,
                64,
            )

        # --- Connect button ---
        button = pygame.Rect(36, 646, WIDTH - 72, 30)
        pygame.draw.rect(screen, (16, 26, 40), button, border_radius=6)
        pygame.draw.rect(screen, STORY_CYAN, button, width=2, border_radius=6)
        draw_text(screen, "PRESS SPACE / ENTER TO CONNECT", button.center, 24, STORY_CYAN, center=True, bold=True)
        draw_text(screen, "ESC TO QUIT", (WIDTH // 2, HEIGHT - 13), 16, STORY_TEXT_DIM, center=True)

        pygame.display.flip()
        clock.tick(FPS)


def draw_fixation_cross(screen):
    x, y = WIDTH // 2, HEIGHT // 2
    pygame.draw.line(screen, FOREGROUND, (x - 22, y), (x + 22, y), width=4)
    pygame.draw.line(screen, FOREGROUND, (x, y - 22), (x, y + 22), width=4)


def draw_calibration_target(screen, direction):
    if direction == "center":
        draw_fixation_cross(screen)
        return
    x = WIDTH // 2 - 310 if direction == "left" else WIDTH // 2 + 310
    pygame.draw.circle(screen, FOREGROUND, (x, HEIGHT // 2), 32)


# --------------------------- Educational helpers ---------------------------

def approximate_ema_cutoff_hz(alpha, sampling_rate=NOMINAL_SAMPLING_RATE_HZ):
    """Approximate the cutoff frequency of an EMA low-pass filter."""

    alpha = float(np.clip(alpha, 1e-6, 0.999999))
    return -np.log(alpha) * sampling_rate / (2.0 * np.pi)


def robust_noise_level(values):
    """Robust spread estimate based on the median absolute deviation."""

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) < 5:
        return float("nan")

    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return 1.4826 * mad


def filter_learning_text(config):
    """Return short educational explanations for the selected settings."""

    baseline_cutoff = approximate_ema_cutoff_hz(
        config.baseline_alpha
    )

    if config.filter_type == "Raw":
        filter_line = (
            "Raw keeps drift and fast noise: fastest, but least protected."
        )
        timing_line = "No digital smoothing delay is added."

    elif config.filter_type == "Highpass":
        filter_line = (
            "Highpass removes the slowly changing baseline estimate."
        )
        timing_line = (
            f"Baseline EMA cutoff is approximately {baseline_cutoff:.2f} Hz."
        )

    elif config.filter_type == "EMA Smooth":
        smoothing_cutoff = approximate_ema_cutoff_hz(
            config.smoothing_alpha
        )
        filter_line = (
            "EMA smoothing suppresses fast fluctuations but adds lag."
        )
        timing_line = (
            f"Approximate smoothing cutoff: {smoothing_cutoff:.1f} Hz."
        )

    else:
        duration_ms = (
            config.moving_average_window
            / NOMINAL_SAMPLING_RATE_HZ
            * 1000.0
        )
        delay_ms = (
            (config.moving_average_window - 1)
            / (2.0 * NOMINAL_SAMPLING_RATE_HZ)
            * 1000.0
        )
        filter_line = (
            "Moving average smooths a fixed window of recent samples."
        )
        timing_line = (
            f"Window: {duration_ms:.0f} ms; approximate delay: {delay_ms:.0f} ms."
        )

    detector_line = (
        "Lower threshold = sensitive but more false commands; "
        "higher threshold = stable but more misses."
    )

    return filter_line, timing_line, detector_line


def draw_live_signal_plot(screen, rectangle, raw_values, filtered_values):
    """Draw a lightweight live raw-versus-filtered preview in Pygame."""

    pygame.draw.rect(
        screen,
        (27, 30, 40),
        rectangle,
        border_radius=10,
    )
    pygame.draw.rect(
        screen,
        (70, 76, 92),
        rectangle,
        width=2,
        border_radius=10,
    )

    draw_text(
        screen,
        "LIVE FILTER PREVIEW",
        (rectangle.centerx, rectangle.top + 23),
        23,
        center=True,
        bold=True,
    )

    draw_text(
        screen,
        "Raw",
        (rectangle.left + 18, rectangle.top + 52),
        19,
        MUTED,
    )
    draw_text(
        screen,
        "Filtered",
        (rectangle.left + 82, rectangle.top + 52),
        19,
        ACCENT,
    )

    if len(raw_values) < 10 or len(filtered_values) < 10:
        draw_text(
            screen,
            "Waiting for EOG samples...",
            rectangle.center,
            23,
            MUTED,
            center=True,
        )
        return

    raw = np.asarray(raw_values, dtype=float)
    filtered = np.asarray(filtered_values, dtype=float)
    count = min(len(raw), len(filtered))
    raw = raw[-count:]
    filtered = filtered[-count:]

    # Removing each trace's median lets the player compare fluctuation and
    # smoothing even when the raw ADC signal has a large DC offset.
    raw = raw - np.median(raw)
    filtered = filtered - np.median(filtered)

    combined = np.concatenate([raw, filtered])
    scale = float(np.percentile(np.abs(combined), 95))
    scale = max(scale, 1.0)

    graph_left = rectangle.left + 15
    graph_right = rectangle.right - 15
    graph_top = rectangle.top + 82
    graph_bottom = rectangle.bottom - 18
    graph_mid = (graph_top + graph_bottom) // 2

    pygame.draw.line(
        screen,
        (70, 76, 92),
        (graph_left, graph_mid),
        (graph_right, graph_mid),
        width=1,
    )

    def points(values):
        x_values = np.linspace(
            graph_left,
            graph_right,
            len(values),
        )
        normalized = np.clip(values / scale, -1.0, 1.0)
        y_values = graph_mid - normalized * (
            (graph_bottom - graph_top) * 0.43
        )
        return [
            (int(x_value), int(y_value))
            for x_value, y_value in zip(x_values, y_values)
        ]

    raw_points = points(raw)
    filtered_points = points(filtered)

    if len(raw_points) >= 2:
        pygame.draw.lines(
            screen,
            MUTED,
            False,
            raw_points,
            width=1,
        )
        pygame.draw.lines(
            screen,
            ACCENT,
            False,
            filtered_points,
            width=2,
        )


def prediction_screen(screen, clock, config):
    """Ask the player to predict the filter's behavior before testing it."""

    options = [
        (
            "fast_noisy",
            "Fast but noisy",
            "I expect quick reactions, but more false commands.",
        ),
        (
            "balanced",
            "Balanced",
            "I expect a compromise between stability and reaction speed.",
        ),
        (
            "smooth_slow",
            "Smooth but slow",
            "I expect fewer false commands, but more delay or misses.",
        ),
    ]

    selected = 1

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return None
                if event.key == pygame.K_UP:
                    selected = (selected - 1) % len(options)
                elif event.key == pygame.K_DOWN:
                    selected = (selected + 1) % len(options)
                elif event.key in (pygame.K_RETURN, pygame.K_SPACE):
                    return options[selected][0]

        screen.fill(BACKGROUND)
        draw_text(
            screen,
            "MAKE A SCIENTIFIC PREDICTION",
            (WIDTH // 2, 85),
            40,
            center=True,
            bold=True,
        )
        draw_text(
            screen,
            f"Selected filter: {config.filter_type}",
            (WIDTH // 2, 135),
            26,
            ACCENT,
            center=True,
        )
        draw_text(
            screen,
            "What do you expect before seeing the result?",
            (WIDTH // 2, 180),
            24,
            MUTED,
            center=True,
        )

        for index, (_, title, description) in enumerate(options):
            y = 250 + index * 115
            rectangle = pygame.Rect(180, y - 28, 640, 88)

            if index == selected:
                pygame.draw.rect(
                    screen,
                    (45, 70, 92),
                    rectangle,
                    border_radius=12,
                )
                pygame.draw.rect(
                    screen,
                    ACCENT,
                    rectangle,
                    width=2,
                    border_radius=12,
                )

            draw_text(screen, title, (220, y - 5), 29, bold=True)
            draw_text(
                screen,
                description,
                (220, y + 30),
                21,
                MUTED,
            )

        draw_text(
            screen,
            "UP/DOWN: choose    ENTER: confirm",
            (WIDTH // 2, 640),
            23,
            center=True,
        )
        pygame.display.flip()
        clock.tick(FPS)


# --------------------------- Filter builder ---------------------------

def adjust_filter_config(config, selected_row, change):
    if selected_row == 0:
        index = FILTER_TYPES.index(config.filter_type)
        config.filter_type = FILTER_TYPES[(index + change) % len(FILTER_TYPES)]
    elif selected_row == 1:
        config.baseline_alpha = float(np.clip(config.baseline_alpha + 0.001 * change, 0.980, 0.999))
    elif selected_row == 2:
        if config.filter_type == "EMA Smooth":
            config.smoothing_alpha = float(np.clip(config.smoothing_alpha + 0.05 * change, 0.40, 0.95))
        elif config.filter_type == "Moving Average":
            config.moving_average_window = int(np.clip(config.moving_average_window + change, 3, 20))
    elif selected_row == 3:
        config.threshold_multiplier = float(np.clip(config.threshold_multiplier + 0.10 * change, 0.50, 2.00))
    elif selected_row == 4:
        config.refractory_ms = int(np.clip(config.refractory_ms + 50 * change, 200, 700))


def filter_builder(screen, clock, config, serial_reader):
    """Interactive builder with a live raw/filtered signal preview."""

    selected_row = 0
    preview_processor = FilterProcessor(config)
    raw_buffer = deque(maxlen=PREVIEW_BUFFER_SAMPLES)
    filtered_buffer = deque(maxlen=PREVIEW_BUFFER_SAMPLES)

    def signature():
        return (
            config.filter_type,
            config.baseline_alpha,
            config.smoothing_alpha,
            config.moving_average_window,
        )

    previous_signature = signature()
    serial_reader.clear()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return None
                if event.key == pygame.K_UP:
                    selected_row = (selected_row - 1) % 5
                elif event.key == pygame.K_DOWN:
                    selected_row = (selected_row + 1) % 5
                elif event.key == pygame.K_LEFT:
                    adjust_filter_config(config, selected_row, -1)
                elif event.key == pygame.K_RIGHT:
                    adjust_filter_config(config, selected_row, 1)
                elif event.key in (pygame.K_RETURN, pygame.K_SPACE):
                    return config

        current_signature = signature()
        if current_signature != previous_signature:
            preview_processor = FilterProcessor(config)
            raw_buffer.clear()
            filtered_buffer.clear()
            previous_signature = current_signature

        for _, raw_value in serial_reader.drain(maximum=1000):
            _, _, filtered_value = preview_processor.process(raw_value)
            raw_buffer.append(float(raw_value))
            filtered_buffer.append(float(filtered_value))

        screen.fill(BACKGROUND)
        draw_text(
            screen,
            "HELMHOLTZ AND THE NOISEPOCALYPSE",
            (WIDTH // 2, 40),
            34,
            center=True,
            bold=True,
        )
        draw_text(
            screen,
            "FILTER LAB",
            (250, 85),
            30,
            ACCENT,
            center=True,
            bold=True,
        )

        if config.filter_type == "EMA Smooth":
            third_label = "Smoothing alpha"
            third_value = f"{config.smoothing_alpha:.2f}"
        elif config.filter_type == "Moving Average":
            third_label = "Moving-average window"
            third_value = f"{config.moving_average_window} samples"
        else:
            third_label = "Additional smoothing"
            third_value = "not used"

        rows = [
            ("Filter type", config.filter_type, "FILTER"),
            ("Baseline alpha", f"{config.baseline_alpha:.3f}", "FILTER"),
            (third_label, third_value, "FILTER"),
            (
                "Threshold multiplier",
                f"{config.threshold_multiplier:.2f}",
                "DETECTOR",
            ),
            (
                "Refractory period",
                f"{config.refractory_ms} ms",
                "DETECTOR",
            ),
        ]

        for index, (label, value, group) in enumerate(rows):
            y = 150 + index * 72
            rectangle = pygame.Rect(35, y - 18, 430, 54)

            if index == selected_row:
                pygame.draw.rect(
                    screen,
                    (45, 70, 92),
                    rectangle,
                    border_radius=10,
                )
                pygame.draw.rect(
                    screen,
                    ACCENT,
                    rectangle,
                    width=2,
                    border_radius=10,
                )

            draw_text(screen, label, (55, y), 24)
            draw_text(
                screen,
                value,
                (430, y),
                24,
                ACCENT,
                center=True,
            )
            draw_text(
                screen,
                group,
                (55, y + 28),
                15,
                MUTED,
            )

        draw_live_signal_plot(
            screen,
            pygame.Rect(500, 105, 465, 345),
            raw_buffer,
            filtered_buffer,
        )

        filter_line, timing_line, detector_line = filter_learning_text(config)

        pygame.draw.rect(
            screen,
            (27, 30, 40),
            pygame.Rect(35, 535, 930, 115),
            border_radius=10,
        )
        draw_text(screen, filter_line, (60, 555), 21)
        draw_text(screen, timing_line, (60, 586), 21, ACCENT)
        draw_text(screen, detector_line, (60, 617), 20, MUTED)

        draw_text(
            screen,
            "UP/DOWN: select   LEFT/RIGHT: change   ENTER: predict and calibrate",
            (WIDTH // 2, 678),
            21,
            center=True,
        )

        pygame.display.flip()
        clock.tick(FPS)

# --------------------------- Calibration ---------------------------

def create_calibration_protocol():
    protocol = [("center", CALIBRATION_CENTER_SECONDS)]
    for _ in range(CALIBRATION_REPETITIONS):
        protocol.extend([
            ("left", CALIBRATION_CUE_SECONDS),
            ("center", CALIBRATION_RETURN_SECONDS),
            ("right", CALIBRATION_CUE_SECONDS),
            ("center", CALIBRATION_RETURN_SECONDS),
        ])
    return protocol


def signed_trial_peak(centered_values):
    """Return the dominant signed response in one calibration trial."""

    values = np.asarray(centered_values, dtype=float)

    if len(values) < 5:
        raise ValueError("Not enough values in a calibration trial.")

    negative = float(np.percentile(values, 10))
    positive = float(np.percentile(values, 90))

    return (
        positive
        if abs(positive) >= abs(negative)
        else negative
    )


def calculate_calibration(calibration_data, threshold_multiplier):
    """
    Calculate thresholds from stable center segments and individual trials.

    Center noise is estimated locally inside each center segment. This avoids
    counting the intentional return-to-center eye movement as noise.
    """

    center_segments = calibration_data["center_segments"]
    left_trial_responses = np.asarray(
        calibration_data["left_trial_responses"],
        dtype=float,
    )
    right_trial_responses = np.asarray(
        calibration_data["right_trial_responses"],
        dtype=float,
    )

    if not center_segments:
        raise ValueError("No stable center segments were recorded.")

    if (
        len(left_trial_responses) < 3
        or len(right_trial_responses) < 3
    ):
        raise ValueError("Not enough left/right calibration trials.")

    center_levels = []
    center_residuals = []

    for segment in center_segments:
        segment = np.asarray(segment, dtype=float)

        if len(segment) < 5:
            continue

        local_center = float(np.median(segment))
        center_levels.append(local_center)
        center_residuals.extend(segment - local_center)

    if not center_levels or len(center_residuals) < 20:
        raise ValueError("Not enough stable center samples.")

    center_level = float(np.median(center_levels))

    center_noise = float(
        np.percentile(
            np.abs(np.asarray(center_residuals, dtype=float)),
            90,
        )
    )

    # Median across repeated trials rejects single blinks or cable artifacts.
    left_response = float(np.median(left_trial_responses))
    right_response = float(np.median(right_trial_responses))

    left_polarity = 1 if left_response >= 0 else -1
    right_polarity = 1 if right_response >= 0 else -1

    noise_floor = max(center_noise * 2.0, 1.0)

    left_threshold = max(
        abs(left_response) * 0.45 * threshold_multiplier,
        noise_floor,
    )

    right_threshold = max(
        abs(right_response) * 0.45 * threshold_multiplier,
        noise_floor,
    )

    neutral_threshold = max(
        center_noise * 1.25,
        min(abs(left_response), abs(right_response)) * 0.12,
        1.0,
    )

    opposite_signs = left_polarity != right_polarity

    strong_enough = (
        min(abs(left_response), abs(right_response))
        >= max(
            MINIMUM_CALIBRATION_RESPONSE,
            center_noise * 1.8,
        )
    )

    valid = opposite_signs and strong_enough

    warning_parts = []

    if not opposite_signs:
        warning_parts.append(
            "Left and right did not have opposite signs."
        )

    if not strong_enough:
        warning_parts.append(
            "Directional response was small compared with stable center noise."
        )

    return CalibrationResult(
        center_level=center_level,
        center_noise=center_noise,
        left_response=left_response,
        right_response=right_response,
        left_polarity=left_polarity,
        right_polarity=right_polarity,
        left_threshold=left_threshold,
        right_threshold=right_threshold,
        neutral_threshold=neutral_threshold,
        valid=valid,
        warning=" ".join(warning_parts),
    )


def run_calibration(screen, clock, serial_reader, processor, config):
    """
    Run calibration while separating stable center periods from transitions.

    The old implementation pooled complete center phases. Those phases also
    contained the intentional return eye movement, which inflated center noise
    and produced thresholds that were much too high.
    """

    serial_reader.clear()
    processor.reset()

    calibration_data = {
        "center_segments": [],
        "left_trial_responses": [],
        "right_trial_responses": [],
    }

    protocol = create_calibration_protocol()
    last_center_level = None

    for phase_index, (direction, duration) in enumerate(
        protocol,
        start=1,
    ):
        phase_start = time.perf_counter()
        phase_samples = []

        while time.perf_counter() - phase_start < duration:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None

                if (
                    event.type == pygame.KEYDOWN
                    and event.key == pygame.K_ESCAPE
                ):
                    return None

            if serial_reader.error is not None:
                raise RuntimeError(
                    f"Serial connection lost: {serial_reader.error}"
                )

            screen.fill(BACKGROUND)

            draw_text(
                screen,
                "CALIBRATION",
                (WIDTH // 2, 70),
                42,
                center=True,
                bold=True,
            )

            instruction = (
                "Look at the center and keep still."
                if direction == "center"
                else f"Move only your eyes {direction.upper()} and hold."
            )

            draw_text(
                screen,
                instruction,
                (WIDTH // 2, 120),
                25,
                MUTED,
                center=True,
            )

            draw_calibration_target(screen, direction)

            draw_text(
                screen,
                f"Phase {phase_index}/{len(protocol)}",
                (WIDTH // 2, HEIGHT - 80),
                23,
                MUTED,
                center=True,
            )

            pygame.display.flip()

            for sample_time, raw_value in serial_reader.drain(
                maximum=1000
            ):
                _, _, filtered = processor.process(raw_value)

                phase_elapsed = sample_time - phase_start

                phase_samples.append(
                    (phase_elapsed, filtered)
                )

            clock.tick(FPS)

        if not phase_samples:
            continue

        elapsed_values = np.asarray(
            [item[0] for item in phase_samples],
            dtype=float,
        )

        signal_values = np.asarray(
            [item[1] for item in phase_samples],
            dtype=float,
        )

        if direction == "center":
            stable_mask = (
                elapsed_values
                >= CALIBRATION_CENTER_SETTLE_SECONDS
            )

            stable_values = signal_values[stable_mask]

            if len(stable_values) >= 5:
                calibration_data["center_segments"].append(
                    stable_values.tolist()
                )

                last_center_level = float(
                    np.median(stable_values)
                )

        else:
            if last_center_level is None:
                continue

            response_mask = (
                (elapsed_values >= CALIBRATION_RESPONSE_START_SECONDS)
                & (elapsed_values <= CALIBRATION_RESPONSE_END_SECONDS)
            )

            response_values = (
                signal_values[response_mask]
                - last_center_level
            )

            if len(response_values) >= 5:
                trial_response = signed_trial_peak(
                    response_values
                )

                calibration_data[
                    f"{direction}_trial_responses"
                ].append(trial_response)

    return calculate_calibration(
        calibration_data,
        config.threshold_multiplier,
    )


def calibration_result_screen(screen, clock, serial_reader, processor, calibration):
    """Show calibration quality and block continuation when it is invalid."""

    while True:
        for _, raw_value in serial_reader.drain(maximum=1000):
            processor.process(raw_value)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return None
                if event.key == pygame.K_r:
                    return "repeat"
                if event.key in (pygame.K_SPACE, pygame.K_RETURN):
                    if calibration.valid:
                        return "start"
                    print("Calibration is invalid. Press R and calibrate again.")

        screen.fill(BACKGROUND)
        draw_text(
            screen,
            "CALIBRATION RESULT",
            (WIDTH // 2, 65),
            42,
            center=True,
            bold=True,
        )
        draw_text(
            screen,
            (
                "Calibration looks usable"
                if calibration.valid
                else "Calibration needs attention"
            ),
            (WIDTH // 2, 120),
            30,
            SUCCESS if calibration.valid else DANGER,
            center=True,
            bold=True,
        )

        rows = [
            ("Center noise", calibration.center_noise),
            ("Left response", calibration.left_response),
            ("Right response", calibration.right_response),
            ("Left threshold", calibration.left_threshold),
            ("Right threshold", calibration.right_threshold),
            ("Neutral threshold", calibration.neutral_threshold),
        ]

        for index, (label, value) in enumerate(rows):
            y = 190 + index * 48
            draw_text(screen, label, (285, y), 26)
            draw_text(
                screen,
                f"{value:.1f}",
                (710, y),
                26,
                ACCENT,
                center=True,
            )

        sign_text = (
            "GOOD: LEFT and RIGHT have opposite signs."
            if calibration.left_polarity != calibration.right_polarity
            else "PROBLEM: LEFT and RIGHT have the same sign."
        )
        draw_text(
            screen,
            sign_text,
            (WIDTH // 2, 500),
            22,
            SUCCESS if calibration.left_polarity != calibration.right_polarity else DANGER,
            center=True,
        )

        if calibration.warning:
            draw_text(
                screen,
                calibration.warning,
                (WIDTH // 2, 545),
                20,
                DANGER,
                center=True,
            )

        footer = (
            "SPACE: controlled filter test   R: recalibrate   ESC: quit"
            if calibration.valid
            else "R: recalibrate   ESC: quit"
        )
        draw_text(
            screen,
            footer,
            (WIDTH // 2, 640),
            23,
            center=True,
        )
        pygame.display.flip()
        clock.tick(FPS)

# --------------------------- Controlled filter test ---------------------------

def create_science_protocol():
    """Create a balanced randomized list of LEFT and RIGHT test trials."""

    if SCIENCE_NUMBER_OF_TRIALS % 2 != 0:
        raise ValueError("SCIENCE_NUMBER_OF_TRIALS must be even.")

    directions = (
        ["left"] * (SCIENCE_NUMBER_OF_TRIALS // 2)
        + ["right"] * (SCIENCE_NUMBER_OF_TRIALS // 2)
    )
    random.Random(SCIENCE_RANDOM_SEED).shuffle(directions)
    return directions


def draw_science_phase(screen, trial_number, phase, target, response):
    screen.fill(BACKGROUND)
    draw_text(
        screen,
        "CONTROLLED FILTER TEST",
        (WIDTH // 2, 65),
        40,
        center=True,
        bold=True,
    )
    draw_text(
        screen,
        f"Trial {trial_number}/{SCIENCE_NUMBER_OF_TRIALS}",
        (WIDTH // 2, 112),
        24,
        MUTED,
        center=True,
    )

    if phase in ("fixation", "return"):
        draw_fixation_cross(screen)
        instruction = "Look at the center and stay relaxed."
    else:
        draw_calibration_target(screen, target)
        instruction = f"LOOK {target.upper()}"

    draw_text(
        screen,
        instruction,
        (WIDTH // 2, 165),
        29,
        ACCENT if phase == "cue" else MUTED,
        center=True,
        bold=phase == "cue",
    )

    if response:
        draw_text(
            screen,
            f"Detected: {response.upper()}",
            (WIDTH // 2, 565),
            27,
            SUCCESS if response == target else DANGER,
            center=True,
        )

    draw_text(
        screen,
        "The computer knows the correct direction and measures accuracy and reaction time.",
        (WIDTH // 2, 640),
        21,
        MUTED,
        center=True,
    )
    pygame.display.flip()


def run_scientific_test(
    screen,
    clock,
    serial_reader,
    processor,
    detector,
    config,
):
    """Measure objective LEFT/RIGHT performance before the free runner."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sample_path = DATA_DIRECTORY / f"filter_test_samples_{timestamp}.csv"
    trial_path = DATA_DIRECTORY / f"filter_test_trials_{timestamp}.csv"

    sample_fields = [
        "time",
        "sample",
        "trial",
        "phase",
        "target",
        "raw",
        "baseline",
        "highpass",
        "filtered",
        "centered",
        "detected_event",
        "filter_type",
    ]
    trial_fields = [
        "trial",
        "target",
        "response",
        "outcome",
        "reaction_time_ms",
        "false_positives_before_cue",
    ]

    protocol = create_science_protocol()
    test_start = time.perf_counter()
    sample_number = 0
    trial_rows = []

    fixation_raw = []
    fixation_highpass = []
    fixation_filtered = []
    reaction_times = []

    serial_reader.clear()
    detector.reset()

    with open(sample_path, "w", newline="") as sample_file, open(
        trial_path,
        "w",
        newline="",
    ) as trial_file:
        sample_writer = csv.DictWriter(sample_file, fieldnames=sample_fields)
        trial_writer = csv.DictWriter(trial_file, fieldnames=trial_fields)
        sample_writer.writeheader()
        trial_writer.writeheader()

        for trial_number, target in enumerate(protocol, start=1):
            false_positives = 0

            # ------------------ Stable center / false-positive period
            phase_start = time.perf_counter()
            detector.reset()

            while time.perf_counter() - phase_start < SCIENCE_FIXATION_SECONDS:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return None, None, None
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        return None, None, None

                draw_science_phase(
                    screen,
                    trial_number,
                    "fixation",
                    target,
                    "",
                )

                for sample_timestamp, raw_value in serial_reader.drain(maximum=1000):
                    relative_time = sample_timestamp - test_start
                    baseline, highpass, filtered = processor.process(raw_value)
                    detected, centered = detector.update(filtered, relative_time)

                    fixation_raw.append(float(raw_value))
                    fixation_highpass.append(float(highpass))
                    fixation_filtered.append(float(filtered))

                    if detected is not None:
                        false_positives += 1

                    sample_number += 1
                    sample_writer.writerow({
                        "time": relative_time,
                        "sample": sample_number,
                        "trial": trial_number,
                        "phase": "fixation",
                        "target": target,
                        "raw": raw_value,
                        "baseline": baseline,
                        "highpass": highpass,
                        "filtered": filtered,
                        "centered": centered,
                        "detected_event": detected or "",
                        "filter_type": config.filter_type,
                    })

                clock.tick(FPS)

            # ------------------ Direction cue / response period
            # Remove any sample that was queued just before the cue appeared,
            # so reaction time starts from a clean visual onset.
            serial_reader.clear()
            detector.reset()
            cue_start = time.perf_counter()
            response = ""
            reaction_time_ms = ""

            while time.perf_counter() - cue_start < SCIENCE_RESPONSE_SECONDS:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return None, None, None
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        return None, None, None

                draw_science_phase(
                    screen,
                    trial_number,
                    "cue",
                    target,
                    response,
                )

                for sample_timestamp, raw_value in serial_reader.drain(maximum=1000):
                    relative_time = sample_timestamp - test_start
                    baseline, highpass, filtered = processor.process(raw_value)
                    detected, centered = detector.update(filtered, relative_time)

                    if response == "" and detected is not None:
                        response = detected
                        reaction_time_ms = max(
                            0.0,
                            (sample_timestamp - cue_start) * 1000.0,
                        )

                    sample_number += 1
                    sample_writer.writerow({
                        "time": relative_time,
                        "sample": sample_number,
                        "trial": trial_number,
                        "phase": "cue",
                        "target": target,
                        "raw": raw_value,
                        "baseline": baseline,
                        "highpass": highpass,
                        "filtered": filtered,
                        "centered": centered,
                        "detected_event": detected or "",
                        "filter_type": config.filter_type,
                    })

                clock.tick(FPS)

            if response == "":
                outcome = "miss"
            elif response == target:
                outcome = "correct"
                reaction_times.append(float(reaction_time_ms))
            else:
                outcome = "wrong"

            row = {
                "trial": trial_number,
                "target": target,
                "response": response,
                "outcome": outcome,
                "reaction_time_ms": reaction_time_ms,
                "false_positives_before_cue": false_positives,
            }
            trial_rows.append(row)
            trial_writer.writerow(row)
            trial_file.flush()

            # ------------------ Return to center
            phase_start = time.perf_counter()
            while time.perf_counter() - phase_start < SCIENCE_RETURN_SECONDS:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return None, None, None
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        return None, None, None

                draw_science_phase(
                    screen,
                    trial_number,
                    "return",
                    target,
                    response,
                )

                for sample_timestamp, raw_value in serial_reader.drain(maximum=1000):
                    relative_time = sample_timestamp - test_start
                    baseline, highpass, filtered = processor.process(raw_value)
                    detected, centered = detector.update(filtered, relative_time)

                    sample_number += 1
                    sample_writer.writerow({
                        "time": relative_time,
                        "sample": sample_number,
                        "trial": trial_number,
                        "phase": "return",
                        "target": target,
                        "raw": raw_value,
                        "baseline": baseline,
                        "highpass": highpass,
                        "filtered": filtered,
                        "centered": centered,
                        "detected_event": detected or "",
                        "filter_type": config.filter_type,
                    })

                clock.tick(FPS)

            sample_file.flush()

    correct = sum(row["outcome"] == "correct" for row in trial_rows)
    wrong = sum(row["outcome"] == "wrong" for row in trial_rows)
    misses = sum(row["outcome"] == "miss" for row in trial_rows)
    false_positives = sum(
        int(row["false_positives_before_cue"])
        for row in trial_rows
    )

    left_trials = [row for row in trial_rows if row["target"] == "left"]
    right_trials = [row for row in trial_rows if row["target"] == "right"]
    left_correct = sum(row["outcome"] == "correct" for row in left_trials)
    right_correct = sum(row["outcome"] == "correct" for row in right_trials)

    raw_noise = robust_noise_level(fixation_raw)
    highpass_noise = robust_noise_level(fixation_highpass)
    filtered_noise = robust_noise_level(fixation_filtered)

    if np.isfinite(raw_noise) and raw_noise > 0:
        noise_reduction = 100.0 * (1.0 - filtered_noise / raw_noise)
    else:
        noise_reduction = float("nan")

    metrics = {
        "science_trials": len(trial_rows),
        "science_correct": correct,
        "science_wrong": wrong,
        "science_misses": misses,
        "science_false_positives": false_positives,
        "science_accuracy_percent": (
            100.0 * correct / len(trial_rows)
            if trial_rows
            else 0.0
        ),
        "science_left_accuracy_percent": (
            100.0 * left_correct / len(left_trials)
            if left_trials
            else 0.0
        ),
        "science_right_accuracy_percent": (
            100.0 * right_correct / len(right_trials)
            if right_trials
            else 0.0
        ),
        "science_median_reaction_ms": (
            float(np.median(reaction_times))
            if reaction_times
            else float("nan")
        ),
        "science_raw_noise": raw_noise,
        "science_highpass_noise": highpass_noise,
        "science_filtered_noise": filtered_noise,
        "science_noise_reduction_percent": noise_reduction,
    }

    return sample_path, trial_path, metrics


def science_test_summary_screen(screen, clock, metrics, prediction):
    """Show objective filter results and allow rebuilding before gameplay."""

    prediction_labels = {
        "fast_noisy": "Fast but noisy",
        "balanced": "Balanced",
        "smooth_slow": "Smooth but slow",
    }

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return None
                if event.key == pygame.K_b:
                    return "rebuild"
                if event.key in (pygame.K_SPACE, pygame.K_RETURN):
                    return "game"

        accuracy = metrics["science_accuracy_percent"]
        reaction = metrics["science_median_reaction_ms"]
        reduction = metrics["science_noise_reduction_percent"]

        if accuracy >= 80 and metrics["science_false_positives"] <= 2:
            result_text = "The filter is reliable enough for the runner."
            result_color = SUCCESS
        elif accuracy >= 60:
            result_text = "Usable, but there is room for improvement."
            result_color = ACCENT
        else:
            result_text = "The filter needs more tuning or a new calibration."
            result_color = DANGER

        screen.fill(BACKGROUND)
        draw_text(
            screen,
            "SCIENTIFIC FILTER REPORT",
            (WIDTH // 2, 65),
            40,
            center=True,
            bold=True,
        )
        draw_text(
            screen,
            result_text,
            (WIDTH // 2, 115),
            25,
            result_color,
            center=True,
            bold=True,
        )

        rows = [
            ("Accuracy", f"{accuracy:.1f} %"),
            ("Correct / wrong / missed", (
                f"{metrics['science_correct']} / "
                f"{metrics['science_wrong']} / "
                f"{metrics['science_misses']}"
            )),
            ("False commands at center", metrics["science_false_positives"]),
            ("Median reaction time", (
                f"{reaction:.0f} ms" if np.isfinite(reaction) else "no correct response"
            )),
            ("Measured noise reduction", (
                f"{reduction:.1f} %" if np.isfinite(reduction) else "not available"
            )),
            ("Your prediction", prediction_labels.get(prediction, prediction)),
        ]

        for index, (label, value) in enumerate(rows):
            y = 185 + index * 58
            draw_text(screen, label, (210, y), 25)
            draw_text(screen, value, (760, y), 25, ACCENT, center=True)

        draw_text(
            screen,
            "SPACE: start runner    B: rebuild filter    ESC: quit",
            (WIDTH // 2, 640),
            23,
            center=True,
        )
        pygame.display.flip()
        clock.tick(FPS)


# --------------------------- Runner ---------------------------

def get_streak_multiplier(streak):
    if streak >= 6:
        return 2.0
    if streak >= 3:
        return 1.5
    return 1.0


def lane_to_x(lane):
    return int(LANE_CENTERS[lane])


def move_lane(current_lane, direction):
    if direction == "left":
        return max(0, current_lane - 1)
    if direction == "right":
        return min(LANE_COUNT - 1, current_lane + 1)
    return current_lane


def draw_road(screen):
    pygame.draw.rect(screen, (28, 31, 42), pygame.Rect(ROAD_LEFT, ROAD_TOP, ROAD_RIGHT - ROAD_LEFT, ROAD_BOTTOM - ROAD_TOP), border_radius=12)
    for divider in range(1, LANE_COUNT):
        x = int(ROAD_LEFT + divider * LANE_WIDTH)
        y = ROAD_TOP + 15
        while y < ROAD_BOTTOM:
            pygame.draw.line(screen, (88, 94, 110), (x, y), (x, min(y + 30, ROAD_BOTTOM)), width=3)
            y += 55


def draw_player_impulse(screen, lane, color):
    """Draw the player as a glowing nerve impulse (clearly 'you')."""

    center = (lane_to_x(lane), PLAYER_Y)
    pygame.draw.circle(screen, color, center, 26, width=3)
    pygame.draw.circle(screen, color, center, 15)
    pygame.draw.circle(screen, FOREGROUND, center, 6)
    # A short upward tail hints at a travelling impulse.
    pygame.draw.line(screen, color, (center[0], center[1] + 18), (center[0], center[1] + 40), width=3)

    rect = pygame.Rect(0, 0, PLAYER_WIDTH, PLAYER_HEIGHT)
    rect.center = center
    return rect


def draw_gate_row(screen, gate):
    """Draw one forced-choice row: clean cyan gate plus red noise artifacts."""

    cell_width = int(LANE_WIDTH) - 18
    y = int(gate["y"])

    for lane in range(LANE_COUNT):
        rect = pygame.Rect(0, 0, cell_width, GATE_HEIGHT)
        rect.center = (lane_to_x(lane), y)

        if lane == gate["goal_lane"]:
            pygame.draw.rect(screen, (18, 58, 70), rect, border_radius=10)
            pygame.draw.rect(screen, ACCENT, rect, width=3, border_radius=10)
            pygame.draw.circle(screen, ACCENT, rect.center, 9)
        else:
            pygame.draw.rect(screen, DANGER, rect, border_radius=10)
            pygame.draw.line(
                screen, FOREGROUND,
                (rect.left + 14, rect.centery), (rect.right - 14, rect.centery),
                width=4,
            )


def draw_onboarding_hint(screen, lane):
    """Label the impulse and explain the controls during the first seconds."""

    overlay = pygame.Surface((WIDTH, 96), pygame.SRCALPHA)
    overlay.fill((10, 14, 22, 205))
    screen.blit(overlay, (0, ROAD_TOP + 6))

    draw_text(screen, "Steer the CYAN gate, dodge the red artifacts",
              (WIDTH // 2, ROAD_TOP + 30), 26, ACCENT, center=True, bold=True)
    draw_text(screen, "Move only your EYES left / right to change lane",
              (WIDTH // 2, ROAD_TOP + 66), 22, FOREGROUND, center=True)

    # Arrow + label pointing at the player impulse.
    px, py = lane_to_x(lane), PLAYER_Y
    draw_text(screen, "THIS IS YOU", (px, py - 78), 22, ACCENT, center=True, bold=True)
    pygame.draw.line(screen, ACCENT, (px, py - 64), (px, py - 34), width=3)
    pygame.draw.polygon(screen, ACCENT, [(px - 6, py - 38), (px + 6, py - 38), (px, py - 28)])


def log_game_event(writer, event_time, event_type, lane_before, lane_after, obstacle_lane, score, streak, lives, details=""):
    writer.writerow({
        "time": event_time,
        "event_type": event_type,
        "lane_before": lane_before,
        "lane_after": lane_after,
        "obstacle_lane": obstacle_lane,
        "score": score,
        "streak": streak,
        "lives": lives,
        "details": details,
    })


def run_game(screen, clock, serial_reader, processor, detector, config):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sample_csv_path = DATA_DIRECTORY / f"eye_runner_samples_{timestamp}.csv"
    event_csv_path = DATA_DIRECTORY / f"eye_runner_events_{timestamp}.csv"

    rng = random.Random(ROUND_RANDOM_SEED)
    serial_reader.clear()
    current_lane = 1
    gates = []
    next_gate_id = 1
    lives = STARTING_LIVES
    score = 0
    streak = 0
    longest_streak = 0
    avoided = 0
    collisions = 0
    eog_commands = 0
    eog_left_commands = 0
    eog_right_commands = 0
    blocked_left_commands = 0
    blocked_right_commands = 0
    keyboard_commands = 0

    last_detected_event = "none"
    last_detected_time = -1e9

    flash_until = -1.0
    flash_color = ACCENT
    next_spawn_time = ONBOARDING_SECONDS
    sample_number = 0
    centered_value = 0.0

    game_start = time.perf_counter()
    previous_frame_time = game_start
    running = True

    sample_fields = [
        "time", "sample", "raw", "baseline", "highpass", "filtered", "centered",
        "detected_event", "detector_state", "current_lane", "score", "streak", "lives",
        "filter_type", "baseline_alpha", "smoothing_alpha", "moving_average_window",
        "threshold_multiplier", "refractory_ms",
    ]
    event_fields = [
        "time", "event_type", "lane_before", "lane_after", "obstacle_lane",
        "score", "streak", "lives", "details",
    ]

    with open(sample_csv_path, "w", newline="") as sample_file, open(event_csv_path, "w", newline="") as event_file:
        sample_writer = csv.DictWriter(sample_file, fieldnames=sample_fields)
        event_writer = csv.DictWriter(event_file, fieldnames=event_fields)
        sample_writer.writeheader()
        event_writer.writeheader()

        while running:
            frame_time = time.perf_counter()
            elapsed = frame_time - game_start
            delta_time = frame_time - previous_frame_time
            previous_frame_time = frame_time

            if elapsed >= GAME_DURATION_SECONDS or lives <= 0:
                break
            if serial_reader.error is not None:
                raise RuntimeError(f"Serial connection lost: {serial_reader.error}")

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key in (pygame.K_LEFT, pygame.K_RIGHT):
                        direction = "left" if event.key == pygame.K_LEFT else "right"
                        lane_before = current_lane
                        current_lane = move_lane(current_lane, direction)
                        keyboard_commands += 1
                        log_game_event(event_writer, elapsed, f"keyboard_{direction}", lane_before, current_lane, "", score, streak, lives)

            for sample_timestamp, raw_value in serial_reader.drain(maximum=1000):
                sample_time = sample_timestamp - game_start
                baseline_value, highpass_value, filtered_value = processor.process(raw_value)
                detected_event, centered_value = detector.update(filtered_value, sample_time)

                if detected_event is not None:
                    lane_before = current_lane
                    lane_after = move_lane(
                        current_lane,
                        detected_event,
                    )

                    current_lane = lane_after
                    eog_commands += 1
                    last_detected_event = detected_event
                    last_detected_time = sample_time

                    if detected_event == "left":
                        eog_left_commands += 1

                        if lane_after == lane_before:
                            blocked_left_commands += 1

                    elif detected_event == "right":
                        eog_right_commands += 1

                        if lane_after == lane_before:
                            blocked_right_commands += 1

                    movement_result = (
                        "lane_changed"
                        if lane_after != lane_before
                        else "blocked_at_edge"
                    )

                    print(
                        f"EOG {detected_event.upper():5s} | "
                        f"lane {lane_before} -> {lane_after} | "
                        f"signal={centered_value:.1f} | "
                        f"{movement_result}"
                    )

                    log_game_event(
                        event_writer,
                        sample_time,
                        f"eog_{detected_event}",
                        lane_before,
                        lane_after,
                        "",
                        score,
                        streak,
                        lives,
                        details=(
                            f"centered_signal={centered_value:.3f};"
                            f"movement={movement_result}"
                        ),
                    )

                sample_number += 1
                sample_writer.writerow({
                    "time": sample_time,
                    "sample": sample_number,
                    "raw": raw_value,
                    "baseline": baseline_value,
                    "highpass": highpass_value,
                    "filtered": filtered_value,
                    "centered": centered_value,
                    "detected_event": detected_event or "",
                    "detector_state": detector.state,
                    "current_lane": current_lane,
                    "score": score,
                    "streak": streak,
                    "lives": lives,
                    "filter_type": config.filter_type,
                    "baseline_alpha": config.baseline_alpha,
                    "smoothing_alpha": config.smoothing_alpha,
                    "moving_average_window": config.moving_average_window,
                    "threshold_multiplier": config.threshold_multiplier,
                    "refractory_ms": config.refractory_ms,
                })

                if sample_number % 100 == 0:
                    sample_file.flush()
                    event_file.flush()

            # Spawn one forced-choice gate row per beat (after the tutorial).
            # The clean gate is biased away from the current lane so that every
            # row needs at least one deliberate LEFT/RIGHT eye movement.
            if elapsed >= next_spawn_time:
                other_lanes = [lane for lane in range(LANE_COUNT) if lane != current_lane]
                goal_lane = rng.choice(other_lanes or list(range(LANE_COUNT)))
                gates.append({"id": next_gate_id, "goal_lane": goal_lane, "y": ROAD_TOP - GATE_HEIGHT, "resolved": False})
                next_gate_id += 1
                log_game_event(event_writer, elapsed, "gate_spawn", current_lane, current_lane, goal_lane, score, streak, lives)
                next_spawn_time += GATE_SPAWN_INTERVAL

            gate_speed = BASE_OBSTACLE_SPEED + SPEED_INCREASE_PER_SECOND * elapsed
            for gate in gates:
                gate["y"] += gate_speed * delta_time

            screen.fill(BACKGROUND)
            draw_road(screen)
            remaining = []

            for gate in gates:
                draw_gate_row(screen, gate)

                # Resolve once, by lane, when the row reaches the player band.
                if not gate["resolved"] and gate["y"] >= PLAYER_Y - GATE_HEIGHT * 0.5:
                    gate["resolved"] = True

                    if current_lane == gate["goal_lane"]:
                        avoided += 1
                        streak += 1
                        longest_streak = max(longest_streak, streak)
                        multiplier = get_streak_multiplier(streak)
                        gained = int(POINTS_PER_AVOIDED_OBSTACLE * multiplier)
                        score += gained
                        flash_until = elapsed + 0.25
                        flash_color = SUCCESS
                        log_game_event(
                            event_writer, elapsed, "gate_passed", current_lane, current_lane,
                            gate["goal_lane"], score, streak, lives,
                            details=f"multiplier={multiplier:.1f};points={gained}",
                        )
                    else:
                        lives -= 1
                        collisions += 1
                        score = max(0, score - COLLISION_PENALTY)
                        streak = 0
                        flash_until = elapsed + 0.25
                        flash_color = DANGER
                        log_game_event(
                            event_writer, elapsed, "gate_missed", current_lane, current_lane,
                            gate["goal_lane"], score, streak, lives,
                        )

                if gate["y"] <= ROAD_BOTTOM + 60:
                    remaining.append(gate)

            gates = remaining

            player_color = flash_color if elapsed < flash_until else ACCENT
            draw_player_impulse(screen, current_lane, player_color)

            if elapsed < ONBOARDING_SECONDS:
                draw_onboarding_hint(screen, current_lane)

            multiplier = get_streak_multiplier(streak)
            remaining_time = max(0.0, GAME_DURATION_SECONDS - elapsed)

            draw_text(screen, f"Score: {score}", (30, 20), 30, bold=True)
            draw_text(screen, f"Lives: {lives}", (30, 55), 26)
            draw_text(screen, f"Streak: {streak}  ×{multiplier:.1f}", (WIDTH - 260, 20), 28)
            draw_text(screen, f"Time: {remaining_time:04.1f} s", (WIDTH - 260, 55), 26)
            draw_text(screen, f"Filter: {config.filter_type}", (WIDTH // 2, 32), 24, ACCENT, center=True)

            # Live EOG diagnostics.
            draw_text(
                screen,
                f"EOG left: {eog_left_commands}",
                (30, 92),
                23,
                SUCCESS,
            )

            draw_text(
                screen,
                f"EOG right: {eog_right_commands}",
                (30, 120),
                23,
                SUCCESS,
            )

            recent_event = (
                elapsed - last_detected_time
                <= 1.0
            )

            draw_text(
                screen,
                f"Last EOG: {last_detected_event.upper()}",
                (WIDTH - 270, 92),
                23,
                SUCCESS if recent_event else MUTED,
            )

            draw_text(
                screen,
                (
                    f"Signal: {centered_value:.0f}   "
                    f"L thr: {detector.calibration.left_threshold:.0f}   "
                    f"R thr: {detector.calibration.right_threshold:.0f}"
                ),
                (WIDTH // 2, 70),
                20,
                MUTED,
                center=True,
            )

            indicator_center_x = WIDTH // 2
            indicator_y = HEIGHT - 18
            half_width = 160
            pygame.draw.line(screen, MUTED, (indicator_center_x - half_width, indicator_y), (indicator_center_x + half_width, indicator_y), width=3)
            signal_scale = max(detector.calibration.left_threshold, detector.calibration.right_threshold, 1.0)
            normalized = float(np.clip(centered_value / (2.0 * signal_scale), -1.0, 1.0))
            indicator_x = int(indicator_center_x + normalized * half_width)
            pygame.draw.circle(screen, ACCENT, (indicator_x, indicator_y), 7)

            pygame.display.flip()
            clock.tick(FPS)

        sample_file.flush()
        event_file.flush()

    game_duration = min(time.perf_counter() - game_start, GAME_DURATION_SECONDS)
    summary = {
        "timestamp": timestamp,
        "filter_type": config.filter_type,
        "baseline_alpha": config.baseline_alpha,
        "smoothing_alpha": config.smoothing_alpha,
        "moving_average_window": config.moving_average_window,
        "threshold_multiplier": config.threshold_multiplier,
        "refractory_ms": config.refractory_ms,
        "random_seed": ROUND_RANDOM_SEED,
        "score": score,
        "game_duration": game_duration,
        "avoided_obstacles": avoided,
        "collisions": collisions,
        "lives_remaining": lives,
        "eog_commands": eog_commands,
        "eog_left_commands": eog_left_commands,
        "eog_right_commands": eog_right_commands,
        "blocked_left_commands": blocked_left_commands,
        "blocked_right_commands": blocked_right_commands,
        "keyboard_commands": keyboard_commands,
        "longest_streak": longest_streak,
        "recorded_samples": sample_number,
    }
    return sample_csv_path, event_csv_path, summary


def show_game_summary(screen, clock, summary, science_metrics, prediction):
    """Show gameplay and controlled-test results together."""

    prediction_labels = {
        "fast_noisy": "Fast but noisy",
        "balanced": "Balanced",
        "smooth_slow": "Smooth but slow",
    }

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
            if event.type == pygame.KEYDOWN and event.key in (
                pygame.K_SPACE,
                pygame.K_RETURN,
                pygame.K_ESCAPE,
            ):
                return

        accuracy = science_metrics["science_accuracy_percent"]
        reaction = science_metrics["science_median_reaction_ms"]
        false_positives = science_metrics["science_false_positives"]

        if accuracy >= 85 and summary["collisions"] <= 1:
            helmholtz_feedback = "Extraordinary! The filter was both accurate and useful in the game."
            feedback_color = SUCCESS
        elif accuracy >= 60:
            helmholtz_feedback = "Good work! Your filter works, but its parameters can still improve."
            feedback_color = ACCENT
        else:
            helmholtz_feedback = "The Noisepocalypse wins this round. Rebuild the filter and compare again."
            feedback_color = DANGER

        screen.fill(BACKGROUND)
        draw_text(
            screen,
            "FINAL LEARNING REPORT",
            (WIDTH // 2, 55),
            40,
            center=True,
            bold=True,
        )
        draw_text(
            screen,
            helmholtz_feedback,
            (WIDTH // 2, 100),
            21,
            feedback_color,
            center=True,
            bold=True,
        )

        draw_text(screen, "CONTROLLED TEST", (270, 155), 27, ACCENT, center=True, bold=True)
        draw_text(screen, "RUNNER", (735, 155), 27, ACCENT, center=True, bold=True)

        left_rows = [
            ("Accuracy", f"{accuracy:.1f} %"),
            ("False positives", false_positives),
            ("Wrong responses", science_metrics["science_wrong"]),
            ("Missed responses", science_metrics["science_misses"]),
            ("Median reaction", (
                f"{reaction:.0f} ms" if np.isfinite(reaction) else "n/a"
            )),
            ("Prediction", prediction_labels.get(prediction, prediction)),
        ]
        right_rows = [
            ("Score", summary["score"]),
            ("Avoided obstacles", summary["avoided_obstacles"]),
            ("Collisions", summary["collisions"]),
            ("EOG left / right", (
                f"{summary['eog_left_commands']} / {summary['eog_right_commands']}"
            )),
            ("Longest streak", summary["longest_streak"]),
            ("Filter", summary["filter_type"]),
        ]

        for index, (label, value) in enumerate(left_rows):
            y = 205 + index * 52
            draw_text(screen, label, (80, y), 23)
            draw_text(screen, value, (405, y), 23, ACCENT, center=True)

        for index, (label, value) in enumerate(right_rows):
            y = 205 + index * 52
            draw_text(screen, label, (535, y), 23)
            draw_text(screen, value, (900, y), 23, ACCENT, center=True)

        draw_text(
            screen,
            "SPACE: save plots and exit",
            (WIDTH // 2, 650),
            23,
            center=True,
        )
        pygame.display.flip()
        clock.tick(FPS)

# --------------------------- Results and plots ---------------------------

def append_round_summary(summary):
    """Append a row while allowing future versions to add more columns."""

    path = RESULT_DIRECTORY / "eye_runner_learning_rounds.csv"
    new_row = pd.DataFrame([summary])

    if path.exists():
        old_rows = pd.read_csv(path)
        all_rows = pd.concat([old_rows, new_row], ignore_index=True, sort=False)
    else:
        all_rows = new_row

    all_rows.to_csv(path, index=False)
    return path

def estimate_sampling_rate(dataframe):
    values = pd.to_numeric(dataframe["time"], errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 10:
        return NOMINAL_SAMPLING_RATE_HZ
    duration = values[-1] - values[0]
    if duration <= 0:
        return NOMINAL_SAMPLING_RATE_HZ
    rate = (len(values) - 1) / duration
    return rate if 20.0 <= rate <= 500.0 else NOMINAL_SAMPLING_RATE_HZ


def resample_to_uniform_grid(time_values, signal_values, sampling_rate):
    time_values = np.asarray(time_values, dtype=float)
    signal_values = np.asarray(signal_values, dtype=float)
    mask = np.isfinite(time_values) & np.isfinite(signal_values)
    time_values = time_values[mask]
    signal_values = signal_values[mask]
    order = np.argsort(time_values)
    time_values = time_values[order]
    signal_values = signal_values[order]
    unique_times, unique_indices = np.unique(time_values, return_index=True)
    signal_values = signal_values[unique_indices]
    if len(unique_times) < 16:
        raise ValueError("Not enough samples for frequency analysis.")
    uniform_time = np.arange(unique_times[0], unique_times[-1], 1.0 / sampling_rate)
    uniform_signal = np.interp(uniform_time, unique_times, signal_values)
    return uniform_time, uniform_signal


def calculate_welch_psd(signal_values, sampling_rate):
    desired_length = int(round(PSD_SEGMENT_DURATION_SECONDS * sampling_rate))
    nperseg = min(len(signal_values), max(16, desired_length))
    frequencies, density = welch(
        signal_values,
        fs=sampling_rate,
        window="hann",
        nperseg=nperseg,
        noverlap=nperseg // 2,
        detrend="constant",
        scaling="density",
    )
    density_db = 10.0 * np.log10(np.maximum(density, POWER_EPSILON))
    return frequencies, density_db


# Matplotlib colours for the dashboard (hex, independent of the pygame UI ints).
DASH_CYAN = "#5ac8ff"
DASH_GREEN = "#5cdc8c"
DASH_RED = "#f0625a"
DASH_ORANGE = "#f0a85a"
DASH_PURPLE = "#c9a0ff"
DASH_GREY = "#8a93a6"


def _bar_labels(axis, bars, fmt="{:.0f}"):
    """Write the numeric value on top of each bar."""

    for bar in bars:
        height = bar.get_height()
        if np.isfinite(height):
            axis.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                fmt.format(height),
                ha="center", va="bottom", fontsize=9,
            )


def _panel_time_domain(axes, samples, events, summary):
    ax_raw, ax_hp, ax_filt = axes
    ax_raw.plot(samples["time"], samples["raw"], color=DASH_GREY, linewidth=0.8)
    ax_raw.set_ylabel("ADC value")
    ax_raw.set_title("TIME DOMAIN — Raw EOG", fontsize=12, fontweight="bold")
    ax_hp.plot(samples["time"], samples["highpass"], color=DASH_CYAN, linewidth=0.8)
    ax_hp.axhline(0, linestyle="--", linewidth=0.8, color="grey")
    ax_hp.set_ylabel("High-pass")
    ax_hp.set_title("Baseline-removed", fontsize=11)
    ax_filt.plot(samples["time"], samples["filtered"], color=DASH_GREEN, linewidth=0.8)
    ax_filt.axhline(0, linestyle="--", linewidth=0.8, color="grey")
    ax_filt.set_ylabel("Filtered")
    ax_filt.set_xlabel("Time [s]")
    ax_filt.set_title(f"Filtered ({summary['filter_type']})", fontsize=11)

    if events is not None and not events.empty:
        commands = events[events["event_type"].isin(["eog_left", "eog_right"])]
        top = ax_filt.get_ylim()[1]
        for _, event in commands.iterrows():
            event_time = float(event["time"])
            label = "L" if event["event_type"] == "eog_left" else "R"
            ax_filt.axvline(event_time, linestyle=":", alpha=0.3, color="grey")
            ax_filt.text(event_time, top, label, fontsize=7, verticalalignment="top")


def _panel_psd(axes, samples, sampling_rate, max_frequency, summary):
    signals = [
        ("FREQUENCY (Welch PSD) — Raw EOG", "raw", True),
        ("High-pass EOG", "highpass", False),
        (f"Filtered ({summary['filter_type']})", "filtered", False),
    ]
    colours = [DASH_GREY, DASH_CYAN, DASH_GREEN]
    for axis, (title, column, is_header), colour in zip(axes, signals, colours):
        try:
            _, uniform_signal = resample_to_uniform_grid(
                samples["time"], samples[column], sampling_rate,
            )
            frequencies, power_db = calculate_welch_psd(uniform_signal, sampling_rate)
            mask = frequencies <= max_frequency
            axis.plot(frequencies[mask], power_db[mask], color=colour, linewidth=1.1)
        except ValueError:
            axis.text(0.5, 0.5, "not enough data", ha="center", va="center", transform=axis.transAxes)
        axis.set_ylabel("PSD [dB/Hz]")
        axis.set_title(title, fontsize=12 if is_header else 11, fontweight="bold" if is_header else "normal")
        axis.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Frequency [Hz]")
    axes[-1].set_xlim(0, max_frequency)


def _panel_runner_bar(axis, summary):
    labels = ["Avoided", "Collisions", "EOG L", "EOG R", "Streak"]
    values = [
        summary["avoided_obstacles"],
        summary["collisions"],
        summary["eog_left_commands"],
        summary["eog_right_commands"],
        summary["longest_streak"],
    ]
    bars = axis.bar(labels, values, color=[DASH_GREEN, DASH_RED, DASH_CYAN, DASH_CYAN, DASH_PURPLE])
    axis.set_ylabel("Count")
    axis.set_title("PERFORMANCE — Runner", fontsize=12, fontweight="bold")
    _bar_labels(axis, bars)


def _panel_outcomes_bar(axis, trials):
    if trials is None or trials.empty:
        axis.axis("off")
        axis.text(0.5, 0.5, "No controlled-test data", ha="center", va="center",
                  transform=axis.transAxes, color="grey")
        return
    counts = [
        int((trials["outcome"] == "correct").sum()),
        int((trials["outcome"] == "wrong").sum()),
        int((trials["outcome"] == "miss").sum()),
        int(pd.to_numeric(trials["false_positives_before_cue"], errors="coerce").fillna(0).sum()),
    ]
    bars = axis.bar(["Correct", "Wrong", "Missed", "False+"], counts,
                    color=[DASH_GREEN, DASH_ORANGE, DASH_RED, DASH_PURPLE])
    axis.set_ylabel("Count")
    axis.set_title("Controlled L/R test outcomes", fontsize=11)
    _bar_labels(axis, bars)


def _panel_noise_bar(axis, fixation):
    if fixation is None or fixation.empty:
        axis.axis("off")
        axis.text(0.5, 0.5, "No fixation data", ha="center", va="center",
                  transform=axis.transAxes, color="grey")
        return
    values = [
        robust_noise_level(pd.to_numeric(fixation["raw"], errors="coerce")),
        robust_noise_level(pd.to_numeric(fixation["highpass"], errors="coerce")),
        robust_noise_level(pd.to_numeric(fixation["filtered"], errors="coerce")),
    ]
    bars = axis.bar(["Raw", "High-pass", "Filtered"], values, color=[DASH_GREY, DASH_CYAN, DASH_GREEN])
    axis.set_ylabel("Robust noise [ADC]")
    axis.set_title("Center noise by stage", fontsize=11)
    _bar_labels(axis, bars, fmt="{:.1f}")


def create_result_plots(
    sample_csv_path,
    event_csv_path,
    summary,
    science_sample_path=None,
    science_trial_path=None,
):
    samples = pd.read_csv(sample_csv_path)
    events = pd.read_csv(event_csv_path)

    for column in ["time", "raw", "baseline", "highpass", "filtered", "centered"]:
        samples[column] = pd.to_numeric(samples[column], errors="coerce")
    samples = samples.dropna(
        subset=["time", "raw", "baseline", "highpass", "filtered", "centered"]
    ).reset_index(drop=True)

    if len(samples) < 16:
        print("Not enough samples to create plots.")
        return

    sampling_rate = estimate_sampling_rate(samples)
    max_frequency = min(PSD_MAX_FREQUENCY_HZ, sampling_rate / 2.0)
    stem = sample_csv_path.stem

    # ------------------ Optional controlled-test data
    trials = None
    if science_trial_path is not None and Path(science_trial_path).exists():
        try:
            trials = pd.read_csv(science_trial_path)
        except (OSError, pd.errors.ParserError):
            trials = None

    fixation = None
    if science_sample_path is not None and Path(science_sample_path).exists():
        try:
            science_samples = pd.read_csv(science_sample_path)
            fixation = science_samples[science_samples["phase"] == "fixation"].copy()
        except (OSError, pd.errors.ParserError):
            fixation = None

    # ------------------ Header metrics line
    runner_total = summary["avoided_obstacles"] + summary["collisions"]
    runner_accuracy = (
        100.0 * summary["avoided_obstacles"] / runner_total
        if runner_total else float("nan")
    )
    metrics = (
        f"Filter: {summary['filter_type']}      "
        f"Score: {summary['score']}      "
        f"Runner hits: {summary['avoided_obstacles']}/{runner_total} ({runner_accuracy:.0f}%)"
    )
    if trials is not None and not trials.empty:
        trial_count = len(trials)
        correct = int((trials["outcome"] == "correct").sum())
        accuracy = 100.0 * correct / trial_count if trial_count else float("nan")
        reaction = pd.to_numeric(
            trials.loc[trials["outcome"] == "correct", "reaction_time_ms"],
            errors="coerce",
        ).dropna()
        reaction_median = float(reaction.median()) if len(reaction) else float("nan")
        metrics += (
            f"      |      Test accuracy: {accuracy:.0f}% ({correct}/{trial_count})"
            f"      Median RT: {reaction_median:.0f} ms"
        )
    if fixation is not None and not fixation.empty:
        raw_noise = robust_noise_level(pd.to_numeric(fixation["raw"], errors="coerce"))
        filtered_noise = robust_noise_level(pd.to_numeric(fixation["filtered"], errors="coerce"))
        if np.isfinite(raw_noise) and raw_noise > 0 and np.isfinite(filtered_noise):
            reduction = 100.0 * (1.0 - filtered_noise / raw_noise)
            metrics += f"      |      Noise reduction: {reduction:.0f}%"

    # ------------------ Assemble one dashboard figure
    figure = plt.figure(figsize=(20, 12))
    grid = figure.add_gridspec(4, 3, height_ratios=[0.42, 1, 1, 1], hspace=0.5, wspace=0.24)

    header_axis = figure.add_subplot(grid[0, :])
    header_axis.axis("off")
    header_axis.text(0.5, 0.7, "Helmholtz and the Noisepocalypse — Result Dashboard",
                     ha="center", va="center", fontsize=20, fontweight="bold")
    header_axis.text(0.5, 0.16, metrics, ha="center", va="center", fontsize=13)

    time_axes = [figure.add_subplot(grid[1, 0])]
    time_axes.append(figure.add_subplot(grid[2, 0], sharex=time_axes[0]))
    time_axes.append(figure.add_subplot(grid[3, 0], sharex=time_axes[0]))
    _panel_time_domain(time_axes, samples, events, summary)

    psd_axes = [figure.add_subplot(grid[1, 1])]
    psd_axes.append(figure.add_subplot(grid[2, 1], sharex=psd_axes[0]))
    psd_axes.append(figure.add_subplot(grid[3, 1], sharex=psd_axes[0]))
    _panel_psd(psd_axes, samples, sampling_rate, max_frequency, summary)

    _panel_runner_bar(figure.add_subplot(grid[1, 2]), summary)
    _panel_outcomes_bar(figure.add_subplot(grid[2, 2]), trials)
    _panel_noise_bar(figure.add_subplot(grid[3, 2]), fixation)

    dashboard_path = PLOT_DIRECTORY / f"{stem}_dashboard.png"
    figure.savefig(dashboard_path, dpi=150)

    print(f"Estimated sampling rate: {sampling_rate:.2f} Hz")
    print(f"Result dashboard: {dashboard_path.resolve()}")

    plt.show()

# --------------------------- Main ---------------------------

def main():
    config = FilterConfig()

    try:
        port = find_serial_port()
        print(f"Opening serial port: {port}")
        serial_connection = serial.Serial(port, BAUD_RATE, timeout=0.2)
    except (FileNotFoundError, RuntimeError, serial.SerialException) as error:
        print(error)
        return

    time.sleep(1.5)
    serial_connection.reset_input_buffer()
    serial_reader = SerialReader(serial_connection)
    serial_reader.start()

    if not test_serial_stream(serial_reader):
        serial_reader.stop()
        serial_connection.close()
        print("Close Thonny and other serial readers, then check code.py and the USB port.")
        return

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Helmholtz and the Noisepocalypse")
    clock = pygame.time.Clock()

    if story_screen(screen, clock) is None:
        serial_reader.stop()
        serial_connection.close()
        pygame.quit()
        return

    game_sample_path = None
    game_event_path = None
    science_sample_path = None
    science_trial_path = None
    game_summary = None
    science_metrics = None
    prediction = None

    try:
        while True:
            selected_config = filter_builder(
                screen,
                clock,
                config,
                serial_reader,
            )
            if selected_config is None:
                return

            prediction = prediction_screen(
                screen,
                clock,
                selected_config,
            )
            if prediction is None:
                return

            processor = FilterProcessor(selected_config)

            while True:
                calibration = run_calibration(
                    screen,
                    clock,
                    serial_reader,
                    processor,
                    selected_config,
                )
                if calibration is None:
                    return

                decision = calibration_result_screen(
                    screen,
                    clock,
                    serial_reader,
                    processor,
                    calibration,
                )
                if decision is None:
                    return
                if decision == "repeat":
                    continue
                break

            detector = EyeMovementDetector(
                calibration,
                selected_config.refractory_ms,
            )

            (
                science_sample_path,
                science_trial_path,
                science_metrics,
            ) = run_scientific_test(
                screen,
                clock,
                serial_reader,
                processor,
                detector,
                selected_config,
            )

            if science_metrics is None:
                return

            science_action = science_test_summary_screen(
                screen,
                clock,
                science_metrics,
                prediction,
            )

            if science_action is None:
                return
            if science_action == "rebuild":
                continue

            detector.reset()
            (
                game_sample_path,
                game_event_path,
                game_summary,
            ) = run_game(
                screen,
                clock,
                serial_reader,
                processor,
                detector,
                selected_config,
            )

            combined_summary = dict(game_summary)
            combined_summary.update(science_metrics)
            combined_summary["prediction"] = prediction

            summary_path = append_round_summary(combined_summary)

            print(f"Game sample data: {game_sample_path.resolve()}")
            print(f"Game events: {game_event_path.resolve()}")
            print(f"Controlled test samples: {science_sample_path.resolve()}")
            print(f"Controlled test trials: {science_trial_path.resolve()}")
            print(f"Combined result table: {summary_path.resolve()}")

            show_game_summary(
                screen,
                clock,
                game_summary,
                science_metrics,
                prediction,
            )
            break

    finally:
        serial_reader.stop()
        serial_connection.close()
        pygame.quit()

    if (
        game_sample_path is not None
        and game_event_path is not None
        and game_summary is not None
    ):
        create_result_plots(
            game_sample_path,
            game_event_path,
            game_summary,
            science_sample_path,
            science_trial_path,
        )



if __name__ == "__main__":
    main()
