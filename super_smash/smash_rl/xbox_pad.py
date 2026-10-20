"""An Xbox pad read in Python and replayed to a Melee port (2 by default) through a libmelee pipe (Linux).

Reading the pad here instead of letting Dolphin map it gives the human player
controls Melee itself doesn't have, without changing anything for the agent:

* The stick never makes you jump, only X and Y do. Pushing up is capped just below
  Melee's jump threshold, except where up means something else: an up-B (while B is
  held or during a special move), a throw, or getting up from the ledge.
* Charged smashes on the right stick (charge_smash, on by default): on the ground, holding the
  right stick in a direction is a smash attack in that direction that keeps charging until the
  stick is let go (sent as the control stick + A held, which is how Melee charges; a plain C-stick
  smash can't be charged). In the air the right stick is the normal C-stick (aerials).
* Optional (smash_hold > 0, off by default): a direction + A held for a moment does a
  smash attack (sent as a C-stick flick). It delays A until the press is decided, which
  makes attacks feel late, so it is off unless asked for.

Every button press reaches the game for at least MIN_PRESS, so a very quick tap that
falls between two of the game's input reads is never lost.
"""

import fcntl
import glob
import os
import select
import struct
import sys
import threading
import time

import melee

EV_KEY, EV_ABS = 0x01, 0x03
BTN_A, BTN_B, BTN_X, BTN_Y, BTN_TL, BTN_TR, BTN_SELECT, BTN_START = 0x130, 0x131, 0x133, 0x134, 0x136, 0x137, 0x13A, 0x13B
ABS_X, ABS_Y, ABS_Z, ABS_RX, ABS_RY, ABS_RZ, ABS_HAT0X, ABS_HAT0Y = 0, 1, 2, 3, 4, 5, 0x10, 0x11
REQUIRED_KEYS = (BTN_A, BTN_B, BTN_X, BTN_Y, BTN_TR, BTN_START)
REQUIRED_AXES = (ABS_X, ABS_Y, ABS_Z, ABS_RX, ABS_RY, ABS_RZ)
EVENT = struct.Struct("llHHi")  # struct input_event on 64-bit Linux
ABSINFO = struct.Struct("6i")  # struct input_absinfo: value, minimum, maximum, fuzz, flat, resolution

B = melee.Button
BUTTONS = {BTN_B: B.BUTTON_B, BTN_X: B.BUTTON_X, BTN_Y: B.BUTTON_Y, BTN_TR: B.BUTTON_Z, BTN_START: B.BUTTON_START}

# Melee's stick scale, which libmelee's controller also uses: 1.0 is a full tilt.
DEADZONE = 0.2875  # smaller tilts are ignored by the game
MAX_UP = 0.6  # tap jump starts at 0.6625; 0.6 still gives up tilts and up aerials
TRIGGER_PRESS = 0.5  # trigger travel that also presses digital L / R (hard shield, air dodge)
SMASH_HOLD = 0.0  # seconds A has to be held with a direction to smash instead of tilt (0: off)
MIN_PRESS = 0.034  # seconds every button press is held for at least (2 frames)
TAP_TIME = 0.035  # how long a tilt's delayed A press is held (about 2 frames)
FLICK_TIME = 0.05  # how long a held-A smash's C-stick flick lasts (3 frames)
CHARGE_START = 0.7  # right-stick tilt that starts a charged smash on the ground
CHARGE_STOP = 0.5  # ... and below which it is released
CHARGE_PREP = 0.017  # one frame of neutral control stick first, so Melee sees a fresh smash flick
POLL = 0.002  # seconds between sends when the pad is quiet
FIRST_CHARACTER_ACTION = 0x155  # action ids from here on are character specific, mostly special moves
FREE_UP_ACTIONS = {melee.Action.GRAB_WAIT, melee.Action.GRAB_PUMMEL,
                   melee.Action.EDGE_CATCHING, melee.Action.EDGE_HANGING}
IN_GAME = (melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH)


def _bitmap_codes(text: str) -> list:
    """Event codes set in a sysfs capability bitmap (hex words, most significant first)."""
    codes = []
    for i, word in enumerate(reversed(text.split())):
        value = int(word, 16)
        codes += [i * 64 + bit for bit in range(64) if value >> bit & 1]
    return codes


def find_pad():
    """(name, /dev/input/eventN) of the first joystick, preferring Xbox pads."""
    pads = []
    for dev in sorted(glob.glob("/sys/class/input/js*/device")):
        events = glob.glob(os.path.join(dev, "event*"))
        if not events:
            continue
        with open(os.path.join(dev, "name")) as f:
            name = f.read().strip()
        with open(os.path.join(dev, "capabilities", "key")) as f:
            keys = _bitmap_codes(f.read())
        with open(os.path.join(dev, "capabilities", "abs")) as f:
            axes = _bitmap_codes(f.read())
        layout_ok = all(c in keys for c in REQUIRED_KEYS) and all(c in axes for c in REQUIRED_AXES)
        pads.append((name, "/dev/input/" + os.path.basename(events[0]), layout_ok))
    if not pads:
        raise SystemExit("No controller found: plug the Xbox controller in by USB, or use --input keyboard")
    pads.sort(key=lambda pad: "box" not in pad[0].lower())
    name, path, layout_ok = pads[0]
    if not layout_ok:
        raise SystemExit(f"'{name}' doesn't have an Xbox pad's buttons and axes (xpad driver); "
                         "use --input keyboard")
    return name, path


def _read_absinfo(fd, code):
    request = (2 << 30) | (ABSINFO.size << 16) | (ord("E") << 8) | (0x40 + code)  # EVIOCGABS(code)
    value, minimum, maximum, *_ = ABSINFO.unpack(fcntl.ioctl(fd, request, bytes(ABSINFO.size)))
    return value, minimum, maximum


class XboxPad:
    """Reads the pad in a thread and drives a Melee port through a libmelee Controller.

    `gamestate` should be kept up to date by whoever steps the console (the human's
    action and on_ground decide what the stick and A do).
    """

    def __init__(self, device: str, smash_hold: float = SMASH_HOLD, absinfo: dict = None, port: int = 2,
                 charge_smash: bool = True):
        """absinfo maps axis code -> (value, minimum, maximum); read from the device when not given."""
        self.smash_hold = smash_hold
        self.charge_smash = charge_smash
        self._charge = None  # (x, y, since) of a right-stick smash being charged
        self.port = port
        self.fd = os.open(device, os.O_RDONLY | os.O_NONBLOCK)
        if absinfo is None:
            absinfo = {code: _read_absinfo(self.fd, code) for code in REQUIRED_AXES}
            for code in (ABS_HAT0X, ABS_HAT0Y):  # the D-pad is optional
                try:
                    absinfo[code] = _read_absinfo(self.fd, code)
                except OSError:
                    pass
        self.absinfo = absinfo
        self.axes = {code: info[0] for code, info in self.absinfo.items()}
        self.keys = set()
        self._pressed_at = {}  # button code -> when it last went down
        self.gamestate = None
        self.controller = None
        self._sent = {}
        self._a_mode = None  # None, "held" (passed through), "pending", "tap" or "smash"
        self._a_down_at = 0.0
        self._tap_until = 0.0
        self._flick = None  # (x, y, until) of a held-A smash
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None

    # --------------------------------------------------------------- lifecycle
    def attach(self, console):
        """Drive our port of a just-started Dolphin whose port is mapped to its libmelee pipe (Pipes/slippibotN)."""
        controller = melee.Controller(console=console, port=self.port, type=melee.ControllerType.STANDARD)
        controller.connect()
        console.controllers.remove(controller)  # flushed by our thread, never by console.step
        with self._lock:
            self.controller = controller
            self._sent = {}

    def start(self):
        if self._thread is None:
            # The bots' networks run in the main thread: hand the GIL over more often so pad events
            # don't wait behind them (Python's default is every 5 ms).
            sys.setswitchinterval(0.0005)
            self._thread = threading.Thread(target=self._run, name="xbox-pad", daemon=True)
            self._thread.start()

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1)
        os.close(self.fd)

    # ------------------------------------------------------------------ input
    def _run(self):
        while not self._stop.is_set():
            ready, _, _ = select.select([self.fd], [], [], POLL)
            if ready:
                try:
                    data = os.read(self.fd, EVENT.size * 64)
                except BlockingIOError:
                    data = b""
                except OSError:
                    print("\nthe controller was disconnected")
                    return
                for i in range(0, len(data) - EVENT.size + 1, EVENT.size):
                    _, _, kind, code, value = EVENT.unpack_from(data, i)
                    self._handle(kind, code, value, time.monotonic())
            self._send(time.monotonic())

    def _handle(self, kind, code, value, now):
        if kind == EV_ABS and code in self.axes:
            self.axes[code] = value
        elif kind == EV_KEY and value in (0, 1):  # 2 is key repeat
            was_down = code in self.keys
            if value:
                self.keys.add(code)
                if not was_down:
                    self._pressed_at[code] = now
            else:
                self.keys.discard(code)
            if code == BTN_A and bool(value) != was_down:
                self._on_a(bool(value), now)

    def _axis(self, code) -> float:
        """-1 .. 1, centred."""
        _, lo, hi = self.absinfo[code]
        return max(-1.0, min(1.0, (2 * self.axes[code] - lo - hi) / (hi - lo)))

    def _trigger(self, code) -> float:
        """0 .. 1."""
        _, lo, hi = self.absinfo[code]
        return max(0.0, min(1.0, (self.axes[code] - lo) / (hi - lo)))

    def _stick(self):
        return self._axis(ABS_X), -self._axis(ABS_Y)  # evdev sticks grow downwards

    def _player(self):
        gs = self.gamestate
        if gs is None or gs.menu_state not in IN_GAME:
            return None
        return gs.players.get(self.port)

    def _up_is_free(self, player) -> bool:
        """Whether pushing up can't start a jump here but means something else."""
        if player is None or BTN_B in self.keys:  # menus, or an up-B
            return True
        action = player.action
        return action in FREE_UP_ACTIONS or getattr(action, "value", 0) >= FIRST_CHARACTER_ACTION

    # ------------------------------------------------------------------- A
    def _on_a(self, down: bool, now: float):
        if down:
            x, y = self._stick()
            player = self._player()
            shielding = max(self._trigger(ABS_Z), self._trigger(ABS_RZ)) >= TRIGGER_PRESS
            self._a_down_at = now
            if (self.smash_hold > 0 and player is not None and player.on_ground
                    and max(abs(x), abs(y)) >= DEADZONE and not shielding and not self._up_is_free(player)):
                self._a_mode = "pending"  # tap or hold? decided in _send
            else:
                self._a_mode = "held"  # straight to the game
        elif self._a_mode == "pending":  # released quickly: the normal attack, A pressed now
            self._a_mode, self._tap_until = "tap", now + TAP_TIME
        elif self._a_mode == "held" and now - self._a_down_at < MIN_PRESS:  # a very quick tap: keep it 2 frames
            self._a_mode, self._tap_until = "tap", self._a_down_at + MIN_PRESS
        elif self._a_mode in ("held", "smash"):
            self._a_mode = None

    def _update_a(self, now: float):
        if self._a_mode == "pending" and now - self._a_down_at >= self.smash_hold:
            x, y = self._stick()
            # Flick the C-stick the way the stick points: sideways is a forward smash.
            flick = (1.0 if x > 0 else -1.0, 0.0) if abs(x) >= abs(y) else (0.0, 1.0 if y > 0 else -1.0)
            self._a_mode, self._flick = "smash", (*flick, now + FLICK_TIME)
        elif self._a_mode == "tap" and now >= self._tap_until:
            self._a_mode = None

    # ---------------------------------------------------------------- output
    def _update_charge(self, c_stick, player, now: float):
        """Start / end a right-stick charged smash; returns (control stick, A) to send while one is on."""
        cx, cy = c_stick
        tilt = max(abs(cx), abs(cy))
        if self._charge is None:
            shielding = max(self._trigger(ABS_Z), self._trigger(ABS_RZ)) >= TRIGGER_PRESS
            if (self.charge_smash and tilt >= CHARGE_START and player is not None and player.on_ground
                    and not shielding):
                direction = (1.0 if cx > 0 else -1.0, 0.0) if abs(cx) >= abs(cy) else (0.0, 1.0 if cy > 0 else -1.0)
                self._charge = (*direction, now)
        elif tilt < CHARGE_STOP:
            self._charge = None
        if self._charge is None:
            return None
        x, y, since = self._charge
        return ((0.0, 0.0), False) if now - since < CHARGE_PREP else ((x, y), True)

    def _send(self, now: float):
        self._update_a(now)
        x, y = self._stick()
        player = self._player()
        c_stick = (self._axis(ABS_RX), -self._axis(ABS_RY))
        charge = self._update_charge(c_stick, player, now)
        if charge is not None:  # the right stick drives the control stick + A; the C-stick stays neutral
            (x, y), charge_a = charge
            c_stick = (0.0, 0.0)
        elif y > MAX_UP and not self._up_is_free(player):
            y = MAX_UP
        if self._flick is not None:
            if now < self._flick[2]:
                c_stick = self._flick[:2]
            else:
                self._flick = None
        l_amount, r_amount = self._trigger(ABS_Z), self._trigger(ABS_RZ)
        hat_x, hat_y = self.axes.get(ABS_HAT0X, 0), self.axes.get(ABS_HAT0Y, 0)
        state = {button: code in self.keys or now - self._pressed_at.get(code, -1.0) < MIN_PRESS
                 for code, button in BUTTONS.items()}
        state.update({
            B.BUTTON_A: self._a_mode in ("held", "tap") or (charge is not None and charge_a),
            B.BUTTON_L: l_amount >= TRIGGER_PRESS,
            B.BUTTON_R: r_amount >= TRIGGER_PRESS,
            B.BUTTON_D_UP: hat_y < 0, B.BUTTON_D_DOWN: hat_y > 0,
            B.BUTTON_D_LEFT: hat_x < 0, B.BUTTON_D_RIGHT: hat_x > 0,
            # Rounded to Melee's stick steps (1/80) so noise doesn't flood the pipe.
            B.BUTTON_MAIN: (round(x * 80) / 80, round(y * 80) / 80),
            B.BUTTON_C: (round(c_stick[0] * 80) / 80, round(c_stick[1] * 80) / 80),
            "L_ANALOG": round(l_amount, 2), "R_ANALOG": round(r_amount, 2),
        })
        with self._lock:
            controller = self.controller
            if controller is None:
                return
            changed = {key: value for key, value in state.items() if self._sent.get(key) != value}
            if not changed:
                return
            for key, value in changed.items():
                if key in (B.BUTTON_MAIN, B.BUTTON_C):
                    controller.tilt_analog_unit(key, *value)
                elif key in ("L_ANALOG", "R_ANALOG"):
                    controller.press_shoulder(B.BUTTON_L if key == "L_ANALOG" else B.BUTTON_R, value)
                elif value:
                    controller.press_button(key)
                else:
                    controller.release_button(key)
            controller.flush()
            self._sent.update(changed)
