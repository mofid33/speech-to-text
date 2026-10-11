# ASR benchmark dataset

مرحله ۶ باید سه مدل را روی **همان dataset، همان device و همان decoding settings** مقایسه کند.

## Manifest

فایل benchmark یک JSONL است؛ هر خط دقیقاً یک نمونه:

```json
{"id":"fa-0001","audio_path":"audio/fa-0001.wav","reference_text":"متن مرجع دقیق فایل صوتی","duration_seconds":31.42,"tags":["noise","names","numbers","technical","mixed_fa_en"]}