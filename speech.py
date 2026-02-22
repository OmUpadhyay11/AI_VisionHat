# speech.py
import time
import threading
import queue
import pyttsx3


class Speaker:
    def __init__(self, rate: int = 175, volume: float = 1.0):
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", rate)
        self.engine.setProperty("volume", volume)

        self.q = queue.Queue()
        self._stop = False
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

        # prevents repeating the same phrase too often
        self.last_text = None
        self.last_time = 0.0

    def _run(self):
        while not self._stop:
            try:
                text = self.q.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                self.engine.say(text)
                self.engine.runAndWait()
            except Exception as e:
                # Don't crash your demo if TTS glitches
                print(f"[TTS] error: {e}")

    def speak(self, text: str, cooldown: float = 1.0, force: bool = False):
        """
        cooldown: minimum seconds between spoken messages
        force: bypass cooldown + same-text suppression (use sparingly)
        """
        now = time.time()

        if not force:
            if text == self.last_text and (now - self.last_time) < max(1.5, cooldown):
                return
            if (now - self.last_time) < cooldown:
                return

        self.last_text = text
        self.last_time = now

        # Drop backlog: keep only the most recent message (better for real-time)
        while not self.q.empty():
            try:
                self.q.get_nowait()
            except queue.Empty:
                break

        self.q.put(text)

    def close(self):
        self._stop = True


speaker = Speaker(rate=180, volume=1.0)