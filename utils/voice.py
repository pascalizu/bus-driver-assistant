import pyttsx3

class VoiceAgent:
    def __init__(self):
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', 150)   # Speaking speed
            self.engine.setProperty('volume', 1.0) # Volume
            self.available = True
        except Exception as e:
            print(f"Voice initialization failed: {e}")
            self.available = False

    def speak(self, text: str):
        """Speak the given text."""
        if self.available:
            print(f"🔊 Announcing: {text}")
            self.engine.say(text)
            self.engine.runAndWait()
        else:
            print(f"🔊 (Voice disabled): {text}")

    def speak_drowsy_alert(self):
        """Speak automatic drowsy driver alert."""
        alert = "Driver, you appear drowsy. For safety, please pull over and take a short break if needed."
        self.speak(alert)

# Global instance
voice_agent = VoiceAgent()