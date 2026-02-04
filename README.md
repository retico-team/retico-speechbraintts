# retico-speechbraintts
Local speechbrain speech synthesis for retico.

## Installation and requirements
You can install the package with the following command:
```bash
pip install git+https://github.com/retico-team/retico-speechbraintts
```

## Example
```python
from retico_core.debug import DebugModule
from retico_core.audio import MicrophoneModule, SpeakerModule
from retico_googleasr import GoogleASRModule
from retico_speechbraintts import SpeechBrainTTSModule


debug = DebugModule(print_payload_only=True)
mic = MicrophoneModule()
asr = GoogleASRModule(rate=16_000)
tts = SpeechBrainTTSModule("en")
speaker = SpeakerModule(rate=22050)

mic.subscribe(asr)
asr.subscribe(tts)
asr.subscribe(debug)
tts.subscribe(speaker)

mic.run()
asr.run()
tts.run()
speaker.run()
debug.run()

input()

mic.stop()
asr.stop()
tts.stop()
speaker.stop()
debug.stop()
```