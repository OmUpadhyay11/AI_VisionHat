import time
import threading
import queue
import pyttsx3


class Speaker:
    def __init__(self, rate: int = 175, volume: float = 1.0, cooldown_sec: float = 1.5):
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", rate)
        self.engine.setProperty("volume", volume)

        self.cooldown_sec = cooldown_sec
        self._last_spoken_at = 0.0
        self._last_text = None

        self.q = queue.Queue()
        self._stop = False

        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while not self._stop:
            try:
                text, force = self.q.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                self.engine.say(text)
                self.engine.runAndWait()
            except Exception as e:
                print("TTS error:", e)

    def say(self, text: str, force: bool = False):
        now = time.time()

        if not force:
            # basic anti-spam
            if (now - self._last_spoken_at) < self.cooldown_sec:
                return
            if text == self._last_text and (now - self._last_spoken_at) < (self.cooldown_sec * 2):
                return

        self._last_spoken_at = now
        self._last_text = text
        self.q.put((text, force))

    def close(self):
        self._stop = True