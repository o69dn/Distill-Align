# Distill-Align

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/o69dn/Distill-Align/actions/workflows/ci.yml/badge.svg)](https://github.com/o69dn/Distill-Align/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/distill-align.svg)](https://pypi.org/project/distill-align/)
[![Security](https://github.com/o69dn/Distill-Align/actions/workflows/security-scan.yml/badge.svg)](https://github.com/o69dn/Distill-Align/actions/workflows/security-scan.yml)

> **Distill-Align: مصنع الاستدلال المنظَّم**
>
> إطار عمل سطر أوامر / بايثون يُؤتمت عملية إنشاء مجموعات بيانات عالية الجودة لضبط النماذج اللغوية (fine-tuning) انطلاقًا من البيانات الخام. يستخدم نماذج الاستدلال المتطورة كمُعلّمين، ويلتقط آثار تفكيرهم العميق، ثم يُرشّحها ويُهذّبها إلى صيغ تعليمية منظمة ومناسبة لضبط النماذج.

## الميزات

- **استيراد ذكي**: أنابيب معالجة غير متزامنة مع تقسيم دلالي لمستندات Markdown و Code (يدعم أيضًا PDF و DOCX و HTML و CSV و JSON و Jupyter notebook وصفحات الويب).
- **توليد عبر مزودات متعددة**: يدعم **OpenAI** و **Ollama** و **vLLM** و **Anthropic Claude** و **Google Gemini** و **Azure OpenAI** مع مجمّعات عمل غير متزامنة.
- **محوّل سقراطي (Socratic Transformer)**: يحوّل الاستدلال الخام إلى حوارات سؤال وجواب متعددة الأدوار ومنظّمة.
- **مُهذّب Scaffold Action**: يزيل الحشو اللغوي لاستخراج المخرجات الهيكلية النقية.
- **تقييم LLM كحَكَم (اختياري)**: تقييم آلي لجودة الحوارات المُنشأة باستخدام نموذج حَكَم منفصل، مع نتائج ثقة من 0 إلى 1.
- **توليد تفضيلات DPO**: إنشاء أزواج تفضيل من الحوارات المُقيّمة لتدريب Direct Preference Optimization.
- **صيغ تصدير متعددة**: ShareGPT، Alpaca، ChatML، HuggingFace messages (JSONL/JSON)، JSON Lines المتدفق، و **Apache Parquet**.
- **تصدير متدفق (Streaming)**: تصدير مجموعات بيانات كبيرة دون تحميلها بالكامل في الذاكرة باستخدام منتجات تكرارية.
- **تتبّع التكاليف**: تقدير التكاليف أثناء الاستخدام عبر جميع المزودات مع محاسبة الرموز لكل طلب.
- **تكامل مع Unsloth**: يولّد نصوص `train.py` مُحسّنة لضبط النماذج باستخدام Unsloth.
- **واجهة طرفية غنية (TUI)**: لوحة تحكم تفاعلية لمراقبة تنفيذ سير العمل.

## التثبيت

```bash
pip install distill-align

# مع الاعتماديات الاختيارية
pip install distill-align[parquet]   # دعم تصدير Parquet
pip install distill-align[hub]       # تكامل مع HuggingFace Hub
pip install distill-align[all]       # جميع الإضافات
```

## البداية السريعة

```bash
# استيراد ومعالجة البيانات
distill-align ingest --source ./my-docs --output ./chunks.json

# توليد الحوارات (مع تقييم الحَكَم)
distill-align synthesize \
    --input ./chunks.json \
    --output ./conversations.json \
    --provider openai \
    --model gpt-4o \
    --judge \
    --judge-model gpt-4o-mini

# التصدير إلى صيغة التدريب
distill-align export \
    --input ./conversations.json \
    --format hf_messages \
    --output ./dataset

# توليد أزواج تفضيل لتدريب DPO
distill-align export \
    --input ./conversations.json \
    --format preference \
    --output ./dpo-pairs

# تشغيل الواجهة التفاعلية
distill-align tui
```

## المزودات المدعومة

| المزود       | بدون SDK | مخرجات منظمة | التوثيق                      |
|--------------|----------|---------------|------------------------------|
| OpenAI       | ✓        | ✓             | مفتاح API                    |
| Anthropic    | ✓        | ✓ (وضع JSON)  | مفتاح API                    |
| Google Gemini| ✓        | ✓ (نوع MIME)  | مفتاح API                    |
| Azure OpenAI | ✓        | ✓             | مفتاح API أو Entra ID (OAuth2) |
| Ollama       | ✓        | —             | لا شيء (محلي)                |
| vLLM         | ✓        | ✓ (متوافق مع OpenAI) | لا شيء / مفتاح API      |

## صيغ التصدير

| الصيغة              | الامتداد      | الوصف                                           |
|---------------------|---------------|-------------------------------------------------|
| `hf_messages`       | `.jsonl`      | صيغة رسائل HuggingFace (JSONL موصى به)          |
| `jsonl`             | `.jsonl`      | JSON Lines عام (يدعم التدفق)                    |
| `parquet`           | `.parquet`    | صيغة عمودية (يتطلب `pyarrow`)                   |
| `sharegpt`          | `.json`       | صيغة محادثات ShareGPT                           |
| `alpaca`            | `.json`       | صيغة تعليمات Alpaca                             |
| `chatml`            | `.json`       | صيغة ترميز ChatML                               |
| `conversation`      | `.json`       | تصدير مخطط المحادثة الخام                        |
| `preference`        | `.json`       | أزواج تفضيل DPO (يتطلب تقييم الحَكَم)           |

## هيكل المشروع

يتبع المشروع نمط **الوحدة الأحادية المعيارية (Modular Monolith)**.

```text
distill-align/
├── src/distill_align/    # حزمة التطبيق الأساسية
│   ├── core/             # الإعدادات، المخططات، التسجيل، التخزين المؤقت، نقاط التفتيش
│   ├── ingestion/        # أدوات تحميل البيانات وتقسيمها (PDF، DOCX، HTML، كود، إلخ)
│   ├── synthesis/        # عملاء LLM، مجمّع العمل، الاستدعاءات، الحَكَم، تتبّع التكاليف
│   │   └── models/       # عملاء خاصون بكل مزود (OpenAI، Anthropic، Gemini، Azure، Ollama، vLLM)
│   ├── exporter/         # أدوات التنسيق، المدقّق، المقسم، مولد التفضيلات
│   │   └── formatters/   # محوّلات صيغ الإخراج (JSONL، Parquet، ShareGPT، Alpaca، إلخ)
│   ├── tui/              # واجهة المستخدم النصية (Textual)
│   └── cli/              # نقاط الدخول عبر Typer
├── tests/                # اختبارات Pytest
└── docs/                 # التوثيق (MkDocs)
```

## التطوير

1. استنساخ المستودع
2. ثبت الاعتماديات بواسطة Poetry: `poetry install`
3. ثبت اعتماديات التطوير: `poetry install --with dev`
4. شغّل الاختبارات: `poetry run pytest`
5. شغّل التدقيق اللغوي: `poetry run ruff check src/`

## الترخيص

رخصة MIT — راجع ملف [LICENSE](LICENSE) للتفاصيل.
