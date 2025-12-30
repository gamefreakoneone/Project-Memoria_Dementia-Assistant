import pyaudio
import wave
import threading
import os
from pydub import AudioSegment

class AudioRecorder:
    def __init__(self):
        self.chunk = 1024
        self.sample_format = pyaudio.paInt16
        self.channels = 1  # Mono for compatibility
        self.fs = 44100
        self.frames = []
        self.is_recording = False
        self.stream = None
        self.p = None
        self.thread = None
        self.current_filename = None
        
    def _record(self):
        """Internal method that runs in a separate thread to record audio."""
        self.p = pyaudio.PyAudio()
        
        try:
            self.stream = self.p.open(
                format=self.sample_format,
                channels=self.channels,
                rate=self.fs,
                frames_per_buffer=self.chunk,
                input=True
            )
            
            print(f"Audio recording started: {self.current_filename}")
            
            while self.is_recording:
                try:
                    data = self.stream.read(self.chunk, exception_on_overflow=False)
                    self.frames.append(data)
                except Exception as e:
                    print(f"Audio read error: {e}")
                    break
                    
        except Exception as e:
            print(f"Audio stream error: {e}")
        finally:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
            if self.p:
                self.p.terminate()
    
    def start_recording(self, filename):
        """
        Start recording audio to the specified filename.
        
        Args:
            filename: Path to save the audio file (should end with .mp3)
        """
        if self.is_recording:
            print("Already recording audio")
            return
            
        self.frames = []
        self.current_filename = filename
        self.is_recording = True
        
        # Start recording in a separate thread
        self.thread = threading.Thread(target=self._record)
        self.thread.start()
    
    def stop_recording(self):
        """Stop recording and save the audio file as MP3."""
        if not self.is_recording:
            print("Not currently recording audio")
            return
            
        self.is_recording = False
        
        # Wait for recording thread to finish
        if self.thread:
            self.thread.join(timeout=2.0)
        
        if not self.frames:
            print("No audio data recorded")
            return
            
        # Save as temporary WAV first
        temp_wav = self.current_filename.replace('.mp3', '_temp.wav')
        
        try:
            # Create a new PyAudio instance for getting sample size
            p_temp = pyaudio.PyAudio()
            sample_width = p_temp.get_sample_size(self.sample_format)
            p_temp.terminate()
            
            # Save WAV temporarily
            wf = wave.open(temp_wav, 'wb')
            wf.setnchannels(self.channels)
            wf.setsampwidth(sample_width)
            wf.setframerate(self.fs)
            wf.writeframes(b''.join(self.frames))
            wf.close()
            
            # Convert to MP3 using pydub
            audio = AudioSegment.from_wav(temp_wav)
            audio.export(self.current_filename, format="mp3", bitrate="128k")
            
            # Remove temporary WAV file
            os.remove(temp_wav)
            
            print(f"Audio saved: {self.current_filename}")
            
        except Exception as e:
            print(f"Error saving audio: {e}")
            # Clean up temp file if it exists
            if os.path.exists(temp_wav):
                os.remove(temp_wav)
        
        self.frames = []
        self.current_filename = None


# For standalone testing
if __name__ == "__main__":
    import time
    
    recorder = AudioRecorder()
    
    # Test recording for 3 seconds
    test_file = "test_audio.mp3"
    recorder.start_recording(test_file)
    print("Recording for 3 seconds...")
    time.sleep(3)
    recorder.stop_recording()
    print("Done!")