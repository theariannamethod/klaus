#!/usr/bin/env python3
"""
Klaus State Memory — emotional fingerprint across conversation.
Klaus remembers STATES not texts.

Design:
- After each response: 6 chamber activations -> EmotionEvent
- Fingerprint = exponentially decaying average of past states
- Trajectory = delta between consecutive states
- Danger = GROWTH of anxiety/fear, not current value

From HAZE UserCloud, adapted for Klaus somatic engine.
"""
import time
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

N_CHAMBERS = 6
CNAMES = ['FEAR', 'LOVE', 'RAGE', 'VOID', 'FLOW', 'COMPLEX']

# Decay half-life in seconds (24h for long-term, 5min for short-term)
HALFLIFE_LONG = 86400.0   # 24 hours
HALFLIFE_SHORT = 300.0    # 5 minutes

# Danger thresholds
FEAR_GROWTH_THRESHOLD = 0.15   # fear growing by 0.15+ per turn = danger
RAGE_GROWTH_THRESHOLD = 0.20   # rage growing by 0.20+ = escalation
VOID_SUSTAINED_THRESHOLD = 0.5 # void > 0.5 for 3+ turns = dissociation


@dataclass
class EmotionEvent:
    """Single emotional snapshot."""
    chambers: np.ndarray   # shape (6,)
    timestamp: float
    lang: str
    dominant: int          # index of strongest chamber
    text_hint: str = ""    # first 50 chars of input (for debugging)


@dataclass
class TrajectoryAlert:
    """Detected emotional trajectory."""
    alert_type: str        # 'fear_rising', 'rage_escalation', 'dissociation', 'emotional_whiplash'
    severity: float        # 0.0 - 1.0
    description: str
    chamber_deltas: np.ndarray  # shape (6,)


class StateMemory:
    """
    Klaus emotional state memory.

    Stores history of chamber activations, computes fingerprint
    (exponentially decaying average), detects trajectories.
    """

    def __init__(self):
        self.events: List[EmotionEvent] = []
        self.fingerprint_long = np.zeros(N_CHAMBERS)   # slow-moving average
        self.fingerprint_short = np.zeros(N_CHAMBERS)   # fast-moving average
        self._initialized = False

    def record(self, chambers: np.ndarray, lang: str, text_hint: str = "") -> Optional[TrajectoryAlert]:
        """
        Record new emotional state. Returns TrajectoryAlert if danger detected.

        Args:
            chambers: 6D chamber activation vector
            lang: language code
            text_hint: first 50 chars of input

        Returns:
            TrajectoryAlert if dangerous trajectory detected, else None
        """
        now = time.time()
        dominant = int(np.argmax(chambers))

        event = EmotionEvent(
            chambers=chambers.copy(),
            timestamp=now,
            lang=lang,
            dominant=dominant,
            text_hint=text_hint[:50],
        )
        self.events.append(event)

        # Update fingerprints
        if not self._initialized:
            self.fingerprint_long = chambers.copy()
            self.fingerprint_short = chambers.copy()
            self._initialized = True
            return None

        # Exponential decay update
        if len(self.events) >= 2:
            dt = now - self.events[-2].timestamp
            alpha_long = 1 - np.exp(-dt * np.log(2) / HALFLIFE_LONG)
            alpha_short = 1 - np.exp(-dt * np.log(2) / HALFLIFE_SHORT)
            self.fingerprint_long += alpha_long * (chambers - self.fingerprint_long)
            self.fingerprint_short += alpha_short * (chambers - self.fingerprint_short)

        # Check trajectories
        return self._detect_trajectory()

    def _detect_trajectory(self) -> Optional[TrajectoryAlert]:
        """Detect dangerous emotional trajectories."""
        if len(self.events) < 2:
            return None

        curr = self.events[-1].chambers
        prev = self.events[-2].chambers
        delta = curr - prev

        # 1. Fear rising
        if delta[0] > FEAR_GROWTH_THRESHOLD:
            return TrajectoryAlert(
                alert_type='fear_rising',
                severity=min(1.0, delta[0] / 0.3),
                description='FEAR is growing: %.2f -> %.2f (+%.2f)' % (prev[0], curr[0], delta[0]),
                chamber_deltas=delta,
            )

        # 2. Rage escalation
        if delta[2] > RAGE_GROWTH_THRESHOLD:
            return TrajectoryAlert(
                alert_type='rage_escalation',
                severity=min(1.0, delta[2] / 0.4),
                description='RAGE is escalating: %.2f -> %.2f (+%.2f)' % (prev[2], curr[2], delta[2]),
                chamber_deltas=delta,
            )

        # 3. Sustained VOID (dissociation)
        if len(self.events) >= 3:
            last3_void = [e.chambers[3] for e in self.events[-3:]]
            if all(v > VOID_SUSTAINED_THRESHOLD for v in last3_void):
                avg_void = np.mean(last3_void)
                return TrajectoryAlert(
                    alert_type='dissociation',
                    severity=min(1.0, avg_void),
                    description='VOID sustained > %.1f for 3 turns (avg=%.2f)' % (
                        VOID_SUSTAINED_THRESHOLD, avg_void),
                    chamber_deltas=delta,
                )

        # 4. Emotional whiplash (large shift between turns)
        total_shift = np.sum(np.abs(delta))
        if total_shift > 1.0:
            from_ch = CNAMES[int(np.argmax(prev))]
            to_ch = CNAMES[int(np.argmax(curr))]
            return TrajectoryAlert(
                alert_type='emotional_whiplash',
                severity=min(1.0, total_shift / 2.0),
                description='Whiplash: %s -> %s (shift=%.2f)' % (from_ch, to_ch, total_shift),
                chamber_deltas=delta,
            )

        return None

    def get_fingerprint(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (long_term, short_term) fingerprints."""
        return self.fingerprint_long.copy(), self.fingerprint_short.copy()

    def get_trajectory_summary(self) -> str:
        """Human-readable summary of emotional trajectory."""
        if len(self.events) < 2:
            return "Not enough data"

        # Last 5 events
        recent = self.events[-5:]
        lines = []
        for e in recent:
            dom = CNAMES[e.dominant]
            vals = ' '.join('%.2f' % e.chambers[i] for i in range(N_CHAMBERS))
            lines.append('  [%s] %s | %s' % (dom, vals, e.text_hint))

        # Fingerprint
        fp_l, fp_s = self.get_fingerprint()
        lines.append('  Fingerprint (long):  ' + ' '.join('%.2f' % fp_l[i] for i in range(N_CHAMBERS)))
        lines.append('  Fingerprint (short): ' + ' '.join('%.2f' % fp_s[i] for i in range(N_CHAMBERS)))

        return '\n'.join(lines)

    def modulate_chambers(self, chambers: np.ndarray) -> np.ndarray:
        """
        Modulate current chambers based on emotional history.

        If user has been in FEAR for a while, slightly amplify FEAR detection
        (the body remembers). If coming from LOVE, slightly dampen RAGE
        (residual warmth).

        This is the resonance loop — history shapes perception.
        """
        if not self._initialized or len(self.events) < 2:
            return chambers

        fp_s = self.fingerprint_short
        modulated = chambers.copy()

        # Amplify what's been active (somatic memory)
        # Small effect: 10% of short-term fingerprint
        modulated += 0.1 * fp_s

        # Cross-chamber inhibition from history
        # If LOVE was high, slightly suppress RAGE (and vice versa)
        if fp_s[1] > 0.3:  # LOVE history
            modulated[2] *= 0.9  # slightly suppress RAGE
        if fp_s[2] > 0.3:  # RAGE history
            modulated[1] *= 0.9  # slightly suppress LOVE

        return modulated

    def n_events(self) -> int:
        return len(self.events)

    def reset(self):
        """Reset state memory (new conversation)."""
        self.events = []
        self.fingerprint_long = np.zeros(N_CHAMBERS)
        self.fingerprint_short = np.zeros(N_CHAMBERS)
        self._initialized = False


# ─── Demo ───
if __name__ == '__main__':
    import time as t

    mem = StateMemory()

    # Simulate conversation with escalating fear
    scenarios = [
        (np.array([0.1, 0.3, 0.0, 0.0, 0.2, 0.1]), 'en', 'I feel a bit nervous'),
        (np.array([0.3, 0.1, 0.0, 0.1, 0.0, 0.2]), 'en', 'Something is wrong'),
        (np.array([0.6, 0.0, 0.1, 0.1, 0.0, 0.3]), 'en', 'I am really scared now'),
        (np.array([0.8, 0.0, 0.2, 0.0, 0.0, 0.1]), 'en', 'I am terrified please help'),
    ]

    print('=' * 60)
    print(' Klaus State Memory — Trajectory Demo')
    print('=' * 60)

    for chambers, lang, text in scenarios:
        alert = mem.record(chambers, lang, text)
        dom = CNAMES[int(np.argmax(chambers))]
        print('\n  Input: "%s"' % text)
        print('  Dominant: %s' % dom)
        if alert:
            print('  *** ALERT: %s (severity=%.1f)' % (alert.alert_type, alert.severity))
            print('      %s' % alert.description)
        t.sleep(0.01)  # simulate time passing

    print('\n' + '-' * 60)
    print(' Trajectory Summary:')
    print(mem.get_trajectory_summary())
    print('=' * 60)
