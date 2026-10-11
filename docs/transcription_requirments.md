# Speech-to-Text Service Requirements

## 1. هدف سرویس

این سرویس برای دریافت فایل صوتی، پردازش asynchronous آن و تولید متن فارسی طراحی می‌شود.

خروجی پردازش باید سه سطح متن را مستقل از یکدیگر نگهداری کند:

1. raw_text
2. normalized_text
3. processed_text

هیچ مرحله‌ای حق overwrite کردن خروجی مرحله قبل را ندارد.

جریان اصلی:

Client
→ FastAPI
→ PostgreSQL / Object Storage
→ RabbitMQ
→ Worker
→ Audio Validation
→ ASR
→ Persian Normalization
→ LLM Post-Processing
→ PostgreSQL


## 2. Runtime Architecture

سیستم یک Codebase و دو Process اجرایی دارد:

### API Process

FastAPI مسئول:

- دریافت درخواست HTTP
- validation ورودی HTTP
- ایجاد Transcription
- upload فایل
- ثبت metadata
- enqueue job
- ارائه status
- ارائه result

Business logic داخل Controller قرار نمی‌گیرد.


### Worker Process

Worker مسئول اجرای pipeline پردازش است:

- دریافت Job
- اعتبارسنجی audio
- اجرای ASR
- ذخیره raw transcript
- normalization
- LLM post-processing
- ذخیره نتیجه
- مدیریت status و failure


## 3. Environmentها

### Local Development

Resources:

- PostgreSQL در Docker
- RabbitMQ در Docker
- MinIO در Docker
- FastAPI روی لپ‌تاپ
- Worker روی لپ‌تاپ
- Faster Whisper روی لپ‌تاپ
- FFmpeg/ffprobe محلی

LLM:

- می‌تواند disabled باشد
- یا از OpenAI-compatible endpoint استفاده کند


### Production

Resources:

- PostgreSQL Server
- RabbitMQ
- S3-compatible Object Storage
- FastAPI Container
- GPU Worker
- FFmpeg/ffprobe
- Faster Whisper / CTranslate2
- Shared vLLM

Speech-to-Text مالک vLLM نیست و فقط consumer آن از طریق HTTP/OpenAI-compatible API است.

تا جای ممکن Local و Production باید یک code path داشته باشند و اختلاف فقط از configuration و resources باشد.


## 4. فایل ورودی

نسخه اول سرویس فقط Audio File را پشتیبانی می‌کند.

Video transcription در scope نسخه اول نیست.


### فرمت‌های مجاز

حداقل فرمت‌های زیر پشتیبانی می‌شوند:

- WAV
- MP3
- M4A
- AAC
- FLAC
- OGG
- OPUS
- WEBM audio


تشخیص اعتبار فایل نباید فقط بر اساس extension باشد.

اعتبار واقعی media در Worker با ffprobe بررسی می‌شود.


## 5. محدودیت فایل

Safe default اولیه:

Maximum file size:

500 MB

Maximum audio duration:

4 hours


این مقادیر باید configuration باشند و داخل business logic hard-code نشوند.


نمونه configuration مفهومی:

transcription:
  upload:
    max_file_size_mb: 500
    max_duration_seconds: 14400


اگر فایل بزرگ‌تر از محدودیت باشد نباید وارد ASR شود.


## 6. زبان

زبان اصلی سرویس:

Persian / fa


سناریوهای زیر باید پشتیبانی شوند:

- فارسی استاندارد
- فارسی محاوره‌ای
- فارسی همراه واژه‌های انگلیسی
- نام اشخاص و شرکت‌ها
- اعداد
- تاریخ
- شماره تلفن
- اصطلاحات فنی انگلیسی در متن فارسی


هدف نسخه اول، Transcription عمومی چندزبانه نیست.

ASR باید برای فارسی بهینه شود، اما وجود واژه انگلیسی داخل گفتار فارسی نباید باعث failure شود.


## 7. ذخیره Audio

Audio binary داخل PostgreSQL ذخیره نمی‌شود.

PostgreSQL فقط metadata را نگه می‌دارد.

فایل اصلی در Object Storage قرار می‌گیرد.

Database باید حداقل reference زیر را نگه دارد:

storage_key


Object Storage:

Local:
MinIO

Production:
S3-compatible Object Storage


## 8. Lifecycle Status

Statusهای اصلی:

created
→ uploaded
→ queued
→ validating
→ transcribing
→ normalizing
→ post_processing
→ completed


Terminal statuses:

failed
cancelled


### معنی Statusها

created:
Transcription record ایجاد شده ولی فایل هنوز آماده پردازش نیست.

uploaded:
Audio در Object Storage ذخیره شده است.

queued:
Job برای Worker ارسال شده است.

validating:
Worker در حال بررسی audio با ffprobe است.

transcribing:
ASR در حال اجرا است.

normalizing:
Persian deterministic normalization در حال اجرا است.

post_processing:
LLM post-processing در حال اجرا یا بررسی است.

completed:
پردازش با موفقیت به پایان رسیده است.

failed:
Pipeline با خطای غیرقابل ادامه متوقف شده است.

cancelled:
Job توسط کاربر یا سیستم لغو شده است.


## 9. State Transition Rules

Status نباید بدون منطق از یک مرحله عبور کند.

Happy path:

created
→ uploaded
→ queued
→ validating
→ transcribing
→ normalizing
→ post_processing
→ completed


Failure می‌تواند از هر مرحله پردازشی رخ دهد:

uploaded
queued
validating
transcribing
normalizing
post_processing

→ failed


completed یک وضعیت terminal است.

Job completed نباید دوباره به‌صورت عادی پردازش شود.

Retry باید operation مستقل و کنترل‌شده باشد.


## 10. Transcript Data

حداقل سه خروجی متن باید ذخیره شود.


### raw_text

خروجی مستقیم ASR.

قواعد:

- immutable از دید مراحل بعد
- normalization روی آن overwrite نمی‌شود
- LLM روی آن overwrite نمی‌شود


### normalized_text

نتیجه Persian deterministic normalization.

ورودی:

raw_text

کارهای مجاز:

- Unicode normalization
- تبدیل ي به ی
- تبدیل ك به ک
- whitespace normalization
- line normalization
- کنترل فاصله
- نیم‌فاصله در قواعد deterministic مشخص


این مرحله اجازه تغییر معنا ندارد.


### processed_text

نتیجه نهایی presentation-level transcript.

ورودی:

normalized_text

LLM فقط اجازه انجام موارد زیر را دارد:

- punctuation
- sentence segmentation
- paragraph segmentation
- اصلاح formatting
- اصلاح نگارشی بسیار محدود و مطمئن


LLM اجازه ندارد:

- summarize کند
- paraphrase کند
- اطلاعات اضافه کند
- اطلاعات حذف کند
- معنی جمله را تغییر دهد
- نام‌ها را حدس بزند
- عددها را تغییر دهد
- اصطلاحات فنی را بدون اطمینان تغییر دهد


## 11. رفتار در Local بدون LLM

LLM باید از configuration قابل فعال/غیرفعال شدن باشد.

در Local می‌توان:

llm.enabled = false

داشت.

در این حالت:

processed_text = normalized_text

اما raw_text و normalized_text همچنان مستقل ذخیره می‌شوند.

همچنین باید مشخص باشد که LLM روی این نتیجه اجرا نشده است.


## 12. Segments

ASR باید علاوه بر raw_text امکان تولید segment داشته باشد.

هر segment حداقل شامل موارد زیر است:

sequence
start_time
end_time
text


Timestampها برحسب ثانیه هستند.

مثال مفهومی:

{
  "sequence": 1,
  "start_time": 0.0,
  "end_time": 4.32,
  "text": "سلام، امروز می‌خواهیم..."
}


ترتیب segmentها باید deterministic باشد.


## 13. API Resource Output

قرارداد نهایی endpointها در مرحله 11 تکمیل می‌شود.

اما resource اصلی Transcription باید حداقل اطلاعات زیر را قابل ارائه کند:

{
  "id": "uuid",
  "status": "transcribing",
  "file_name": "meeting.m4a",
  "content_type": "audio/mp4",
  "file_size": 12345678,
  "duration": 125.4,
  "created_at": "...",
  "updated_at": "...",
  "error": null
}


بعد از completed:

{
  "id": "uuid",
  "status": "completed",
  "raw_text": "...",
  "normalized_text": "...",
  "processed_text": "...",
  "segments": [...]
}


API نباید storage credential یا مسیر داخلی واقعی Object Storage را expose کند.


## 14. Error Contract

خطاهای business/pipeline باید machine-readable error code داشته باشند.

حداقل error codeها:

invalid_file_type
file_too_large
invalid_media
audio_duration_exceeded
audio_validation_failed
storage_error
queue_error
asr_error
normalization_error
llm_timeout
llm_error
processing_error


خطا باید حداقل شامل موارد زیر باشد:

{
  "code": "invalid_media",
  "message": "Uploaded file is not a valid audio file."
}


اطلاعات حساس یا stack trace نباید از API به Client داده شود.


## 15. Failure Policy

اگر Storage قبل از queue شکست بخورد:

Transcription نباید queued شود.


اگر RabbitMQ enqueue شکست بخورد:

وضعیت نباید به queued موفق تبدیل شود.


اگر Audio Validation شکست بخورد:

ASR اجرا نمی‌شود.


اگر ASR شکست بخورد:

normalized_text و processed_text ساخته نمی‌شوند.


اگر Normalization شکست بخورد:

raw_text باید حفظ شود.


اگر LLM شکست بخورد:

raw_text و normalized_text نباید حذف یا overwrite شوند.


رفتار retry دقیق در مرحله Reliability طراحی می‌شود.


## 16. ASR Engine

ASR engine اولیه:

faster-whisper / CTranslate2


هیچ مدل ASR قبل از benchmark به عنوان مدل نهایی اعلام نمی‌شود.


حداقل candidateها:

- openai/whisper-large-v3
- openai/whisper-large-v3-turbo
- nezamisafa/whisper-persian-v4


انتخاب مدل باید صرفاً بر اساس benchmark انجام شود.


## 17. ASR Benchmark Metrics

برای هر مدل حداقل معیارهای زیر اندازه‌گیری می‌شوند:

- WER
- CER
- normalized WER
- normalized CER
- Real-Time Factor (RTF)
- total latency
- RAM usage
- VRAM usage


کیفیت روی categoryهای زیر جداگانه بررسی می‌شود:

- clean speech
- noisy speech
- conversational Persian
- names
- numbers
- technical terminology
- Persian/English mixed speech
- compressed audio


## 18. Benchmark Dataset

Safe default برای اولین benchmark:

حداقل:

10 ساعت صدای واقعی فارسی

و حداقل:

400 نمونه صوتی


Dataset نباید فقط شامل audio تمیز studio باشد.


Target distribution:

30% clean/general Persian

20% noisy speech

10% compressed/mobile/telephone-like audio

15% technical/specialized speech

10% names and numbers intensive

15% Persian-English mixed speech


Dataset باید شامل چند speaker و شرایط ضبط مختلف باشد.


## 19. Benchmark Ground Truth

هر فایل benchmark باید transcript انسانی معتبر داشته باشد.

دو reference می‌تواند نگهداری شود:

reference_raw

reference_normalized


reference_normalized باید با همان قواعد deterministic normalization پروژه تولید شود.

به این ترتیب:

WER/CER

و

normalized WER/CER

هر دو قابل محاسبه هستند.


## 20. Benchmark Fairness

همه مدل‌ها باید تحت شرایط یکسان benchmark شوند:

- hardware یکسان
- dataset یکسان
- preprocessing یکسان
- decoding configuration ثبت‌شده
- batch strategy یکسان تا حد امکان
- compute type ثبت‌شده
- library version ثبت‌شده


نتایج benchmark بدون ثبت environment معتبر محسوب نمی‌شوند.


## 21. معیار انتخاب مدل

هیچ candidate از قبل winner نیست.

انتخاب باید بر اساس trade-off واقعی انجام شود.

اولویت‌ها:

1. accuracy فارسی
2. حفظ names/numbers/technical terms
3. mixed Persian-English quality
4. latency
5. RTF
6. VRAM/RAM usage


امکان دارد مدل انتخابی Local و Production متفاوت باشد.

مثلاً محدودیت سخت‌افزار لپ‌تاپ می‌تواند باعث انتخاب configuration متفاوت Local شود.

این تصمیم فقط بعد از benchmark گرفته می‌شود.


## 22. Service Success Criteria

نسخه اولیه زمانی از نظر functional موفق محسوب می‌شود که:

1. فایل معتبر در Object Storage ذخیره شود.

2. audio binary داخل PostgreSQL ذخیره نشود.

3. Job از API تا Worker قابل ردیابی باشد.

4. فایل نامعتبر قبل از ASR reject شود.

5. raw_text بدون overwrite ذخیره شود.

6. normalized_text مستقل از raw_text ذخیره شود.

7. processed_text مستقل ذخیره شود.

8. segment timestampها قابل دریافت باشند.

9. failure هر مرحله error code مشخص داشته باشد.

10. failure مراحل بعد raw outputهای قبلی را از بین نبرد.

11. job completed دوباره ناخواسته پردازش نشود.

12. Local بدون dependency به vLLM production قابل اجرا باشد.

13. Production بتواند Shared vLLM را از HTTP/OpenAI-compatible API مصرف کند.

14. ASR model از configuration خوانده شود.

15. LLM model از configuration خوانده شود.

16. مدل ASR فقط بعد از benchmark انتخاب شود.


## 23. Performance Success Criteria

در مرحله 0 مقدار ثابت latency برای Production تعیین نمی‌شود، چون hardware GPU نهایی هنوز benchmark نشده است.

Performance باید با metrics واقعی ثبت شود:

- upload latency
- queue wait time
- validation latency
- ASR latency
- normalization latency
- LLM latency
- total processing latency
- RTF


شرط پذیرش performance در مرحله benchmark این است که برای هر candidate اعداد فوق ثبت و قابل مقایسه باشند.

انتخاب مدل فقط براساس accuracy انجام نمی‌شود.


## 24. Reliability Requirements

Pipeline باید در مراحل بعد برای موارد زیر طراحی شود:

- duplicate job
- Worker restart
- RabbitMQ temporary failure
- Storage temporary failure
- ASR failure
- vLLM timeout
- vLLM unavailable
- database failure


raw_text و normalized_text در failure مرحله LLM نباید از دست بروند.


## 25. Security Requirements

حداقل الزامات اولیه:

- file extension به تنهایی trusted نیست
- content/media validation الزامی است
- filename نباید به عنوان storage path trusted استفاده شود
- storage_key توسط server ساخته می‌شود
- Object Storage credential به client داده نمی‌شود
- stack trace به client expose نمی‌شود
- upload size محدود است
- duration محدود است


جزئیات امنیت upload در مرحله 12 تکمیل می‌شود.


## 26. Configuration Requirements

حداقل موارد زیر باید configurable باشند:

- maximum file size
- maximum duration
- allowed formats
- object storage configuration
- RabbitMQ configuration
- ASR model
- ASR device
- ASR compute type
- LLM enabled
- LLM base URL
- LLM model
- LLM timeout


هیچ نام مدل ASR یا LLM نباید داخل business logic hard-code شود.


## 27. Out of Scope برای نسخه اولیه

موارد زیر فعلاً جزو scope نیستند:

- real-time streaming transcription
- WebSocket transcription
- speaker diarization
- speaker identification
- translation
- summarization
- video transcription
- arbitrary multilingual transcription
- editing audio
- storing audio binary in PostgreSQL
- hosting dedicated vLLM inside this service


افزودن هرکدام از این موارد نیازمند requirement مستقل است.


## 28. Acceptance Checklist — Stage 0

- [ ] معماری یک Codebase و دو Process مشخص است.
- [ ] Local environment مشخص است.
- [ ] Production environment مشخص است.
- [ ] Object Storage strategy مشخص است.
- [ ] Audio در PostgreSQL ذخیره نمی‌شود.
- [ ] فرمت‌های ورودی اولیه مشخص هستند.
- [ ] Maximum file size مشخص است.
- [ ] Maximum duration مشخص است.
- [ ] زبان اصلی فارسی مشخص است.
- [ ] mixed Persian/English در scope است.
- [ ] lifecycle statusها مشخص هستند.
- [ ] terminal statusها مشخص هستند.
- [ ] raw_text مشخص شده است.
- [ ] normalized_text مشخص شده است.
- [ ] processed_text مشخص شده است.
- [ ] overwrite نکردن raw_text الزام شده است.
- [ ] Segment contract مشخص شده است.
- [ ] Error codeهای اصلی مشخص شده‌اند.
- [ ] رفتار اولیه failure مشخص شده است.
- [ ] ASR engine اولیه مشخص شده است.
- [ ] سه candidate اصلی benchmark مشخص شده‌اند.
- [ ] معیارهای WER/CER مشخص شده‌اند.
- [ ] normalized WER/CER مشخص شده‌اند.
- [ ] RTF و resource metrics مشخص شده‌اند.
- [ ] dataset benchmark اولیه تعریف شده است.
- [ ] هیچ مدل ASR قبل از benchmark winner اعلام نشده است.
- [ ] LLM در Local قابل disable است.
- [ ] Production از Shared vLLM استفاده می‌کند.
- [ ] LLM model configurable است.
- [ ] ASR model configurable است.
- [ ] LLM حق summarization/paraphrase ندارد.
- [ ] scope و out-of-scope نسخه اول مشخص است.