# retico-speechbraintts
Local speechbrain speech synthesis for retico.

## Installation and Requirements

To use the SpeechBrain TTS module you first need to install the retico-speechbraintts package:
* Install retico-speechbraintts:
```pip install retico-speechbraintts```

  or, for the latest development version from GitHub:
```pip install git+https://github.com/retico-team/retico-speechbraintts.git```

The voice models are downloaded from Hugging Face the first time the module is used.

## Modules

### `SpeechBrainTTSModule`
Subscribes to `TextIU` and outputs `AudioIU` with the synthesized speech at 22050 Hz, in chunks of `frame_duration` seconds. Synthesized speech is cached on disk, so repeated texts are not synthesized again.

#### Arguments:
* `language` (str): the language of the voice; only `"en"` is available; defaults to `"en"`
* `dispatch_on_finish` (bool): if `True`, speech is only synthesized once the input text is committed; if `False`, it is also synthesized while the text is still growing; defaults to `True`
* `frame_duration` (float): the length in seconds of each audio chunk sent forward; defaults to `0.2`

## Example
**Note:** The example uses `MicrophoneModule`, `SpeakerModule` and `DebugModule` from `retico_core` and `GoogleASRModule` from [retico-googleasr](https://github.com/retico-team/retico-googleasr) (`pip install retico-googleasr`) to repeat what you say.

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
