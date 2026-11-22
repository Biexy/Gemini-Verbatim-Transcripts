# مُفرِّغ المحاضرات الذكي — Lecture Transcriber

أداة خفيفة وفائقة الاعتمادية لتحميل وتفريغ المحاضرات الصوتية والمرئية من **YouTube** و **Google Drive** (فيديوهات، قوائم تشغيل، ملفات، ومجلدات) إلى ملفات **Markdown** عربية دقيقة باستخدام **Gemini AI**، مع دعم كامل للاستئناف التلقائي عند انقطاع الإنترنت.

Lightweight, zero-bloat local Web UI & CLI app for downloading and verbatim-transcribing Arabic/academic lectures from **YouTube** and **Google Drive** using **Gemini AI**, built for slow or unstable internet connections.

---

## المميزات الرئيسية / Key Features

- **خفيف وسريع جداً**: خادم ويب محلي مدمج باستخدام مكتبة Python القياسية فقط (`ThreadingHTTPServer`) بدون Flask أو Electron. يبدأ في أقل من 0.2 ثانية ويعمل بواجهة عربية كاملة بدون الحاجة لملفات CDN خارجية.
- **مقاوم لانقطاع الإنترنت**: يكتشف انقطاع الاتصال تلقائياً، ينتظر عودة الإنترنت دون إفشال المهام، ويستأنف التحميل من نفس البايت (`HTTP Range` / `.part`) تلقائياً حتى بعد إغلاق البرنامج.
- **دعم شامل للمصادر**: روابط YouTube الفردية وShorts وPlaylists، وروابط Google Drive الفردية والمجلدات الكاملة.
- **تفريغ عربي دقيق**: تقسيم تلقائي للصوت الطويل مع ضبط التوقيتات المستمرة `[MM:SS]`، وتنظيف تلقائي لحلقات التكرار (Hallucination loops).
- **أوضاع توفير البيانات وحفظ الصوت (Bandwidth Saver & Keep Audio)**:
  - `eco` *(الافتراضي — توفير أقصى)*: يفضّل البث الصوتي الخفيف فقط على يوتيوب، ويجلب النسخ الموفّرة من Google Drive عند توفرها، ويرفع الصوت النقي مباشرةً إلى Gemini بدون إعادة ترميز عند تطابق الشروط، أو يحوّله إلى MP3 خفيف `32k` (~14MB/ساعة).
  - `balanced` *(متوازن)*: يختار جودة صوت أعلى (`64k`) مع بقاء الصوت أولاً والبديل المرئي الطارئ في النهاية فقط.
  - `strict_audio` *(صوت فقط صارم)*: يرفض أي تحميل فيديو على يوتيوب نهائياً.
  - **الاحتفاظ بملفات الصوت (`keep_audio`)**: خيار لحفظ نسخة MP3 خفيفة من المحاضرة داخل `<Course>/audio/<Title>.mp3` بجانب التفريغ.
- **مخرجات جاهزة لـ NotebookLM**: يحفظ كل محاضرة كملف `.md` منفصل، بالإضافة إلى ملف موحّد `000_Complete_<Course>_NotebookLM.md` وتقرير ملخص `_Summary_Report.md`.

---

## التشغيل السريع / Quick Start

### 1. المتطلبات / Prerequisites
- **Python 3.10+**
- **FFmpeg** (مثبّت ومضاف إلى `PATH`)
- **Gemini API Key** (يمكنك إدخاله من زر «الإعدادات» داخل الواجهة أو عبر متغير البيئة `GEMINI_API_KEY`)

### 2. التثبيت / Installation
```bash
cd lecture_transcriber
pip install -r requirements.txt
```

### 3. تشغيل الواجهة الرسومية (Web UI)
- **على ويندوز (بنقرة مزدوجة)**: اضغط مرتين على `Launch_Transcriber.bat`
- **أو من الطرفية**:
```bash
python run.py
```
سيفتح المتصفح تلقائياً على `http://127.0.0.1:8765`.

خيارات إضافية لخادم الويب:
```bash
python run.py --port 8765 --host 127.0.0.1 --no-browser
```

---

## التشغيل من سطر الأوامر (Headless CLI)

يمكنك التفريغ مباشرة من الطرفية مع شريط تقدم حي:

```bash
# تمرير روابط مباشرة
python run.py --cli --course "علوم القرآن" --urls "https://youtu.be/VIDEO1" "https://drive.google.com/file/d/FILE_ID/view"

# أو قراءة الروابط من ملف نصي (رابط في كل سطر)
python run.py --cli --course "علوم القرآن" --file links.txt

# أو استدعاء وحدة CLI مباشرة مع وضع التوفير وحفظ ملفات الصوت
python -m lecture_transcriber.cli --course "اسم المادة" --file links.txt --mode eco --keep-audio --output "C:\Path\To\Transcripts"
```

عرض جميع الخيارات:
```bash
python run.py --help
python -m lecture_transcriber.cli --help
```

---

## واجهة برمجة التطبيقات المحلية / Local REST & SSE API

| Method | Endpoint | الوصف / Description |
|---|---|---|
| `GET` | `/` | واجهة الويب العربية (Single-file HTML) |
| `GET` | `/api/state` | الحالة الكاملة (الإعدادات مع إخفاء المفتاح، المواد، المهام، حالة الإنترنت وFFmpeg) |
| `POST` | `/api/enqueue` | إضافة روابط جديدة `{ "course_name": "...", "urls_text": "..." }` |
| `POST` | `/api/jobs/<id>/pause` | إيقاف مهمة مؤقتاً مع حفظ ملف التحميل الجزئي |
| `POST` | `/api/jobs/<id>/resume` | استئناف مهمة متوقفة |
| `POST` | `/api/jobs/<id>/retry` | إعادة محاولة مهمة فاشلة أو ملغاة |
| `POST` | `/api/jobs/<id>/cancel` | إلغاء مهمة وتنظيف ملفاتها المؤقتة |
| `DELETE` | `/api/jobs/<id>` | حذف مهمة من القائمة |
| `POST` | `/api/pause_all` | إيقاف جميع المهام النشطة مؤقتاً |
| `POST` | `/api/resume_all` | استئناف جميع المهام المتوقفة |
| `GET` | `/api/jobs/<id>/transcript` | جلب نص التفريغ Markdown للمعاينة والنسخ |
| `POST` | `/api/settings` | تحديث الإعدادات (`download_mode`، `keep_audio`، API Key، مجلد الحفظ، عدد العمال، الجودة، البروكسي) |
| `POST` | `/api/open_folder` | فتح مجلد التفريغات في Windows Explorer |
| `GET` | `/api/events` | بث مباشر (Server-Sent Events) للتقدم والسرعة وحالة الشبكة |

---

## استكشاف الأخطاء / Troubleshooting

1. **ظهور تحذير `FFmpeg غير مثبت`**:
   - ثبّت FFmpeg عبر `winget install Gyan.FFmpeg` ثم أعد فتح نافذة الطرفية أو البرنامج.
2. **ملف Google Drive يرفض التحميل (ملف خاص)**:
   - تأكد أن إعدادات مشاركة الملف أو المجلد في Google Drive مضبوطة على **Anyone with the link (Viewer)**.
3. **بطء أو انقطاع الإنترنت أثناء التحميل أو التفريغ**:
   - لا تغلق التطبيق أو أغلقْه بأمان؛ التطبيق يحفظ ملفات `.part` وحالة المهام في `data/state.db` ويستأنف تلقائياً عند عودة الاتصال أو عند التشغيل التالي.
4. **خطأ `429` أو انتهاء حصة Gemini**:
   - ينتقل المحرك تلقائياً بين النماذج البديلة المهيأة في الإعدادات مع تراجع زمني ذكي (backoff). يمكنك أيضاً تقليل «عدد التفريغات المتزامنة» من الإعدادات إلى `1`.
