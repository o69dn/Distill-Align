# Distill-Align

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/o69dn/Distill-Align/actions/workflows/ci.yml/badge.svg)](https://github.com/o69dn/Distill-Align/actions/workflows/ci.yml)

> **Distill-Align: مصنع الاستدلال المنظَّم**  
> إطار عمل يساعدك على إنشاء مجموعات بيانات عالية الجودة لضبط النماذج اللغوية (fine-tuning) انطلاقًا من البيانات الخام.

## الميزات

- **استيراد ذكي** — أنابيب معالجة غير متزامنة مع تقسيم دلالي للمستندات (Markdown و Code).
- **توليد عبر مزودات متعددة** — يدعم OpenAI و Ollama و vLLM مع مجمّعات عمل غير متزامنة.
- **محوّل سقراطي (Socratic Transformer)** — يحوّل الاستدلال الخام إلى حوارات متعددة الأدوار ومنظّمة.
- **مُهذّب Scaffold Action** — يزيل الحشو اللغوي لاستخراج المخرجات الهيكلية النقية.
- **تكامل مع Unsloth** — يولّد نصوص `train.py` مُحسّنة ويصدر إلى صيغ ShareGPT و Alpacha.
- **واجهة طرفية غنية (TUI)** — لوحة تحكم تفاعلية لمراقبة سير العمل.

## التثبيت

```bash
pip install distill-align
```

## البداية السريعة

```bash
# استيراد ومعالجة البيانات
distill-align ingest --source ./my-docs --output ./chunks.json

# توليد الحوارات
distill-align synthesize --input ./chunks.json --output ./conversations.json

# التصدير إلى صيغة التدريب
distill-align export --input ./conversations.json --format sharegpt --output ./dataset

# تشغيل الواجهة التفاعلية
distill-align tui
```

## هيكل المشروع

يتبع المشروع نمط **الوحدة الأحادية المعيارية (Modular Monolith)**.

```text
distill-align/
├── src/distill_align/    # حزمة التطبيق الأساسية
│   ├── core/             # الإعدادات، المخططات، التسجيل
│   ├── ingestion/        # أدوات تحميل البيانات وتقسيمها
│   ├── synthesis/        # عميل LLM، مجمّع العمل، الاستدعاءات
│   ├── exporter/         # أدوات التنسيق وبناء Unsloth
│   ├── tui/              # واجهة المستخدم النصية
│   └── cli/              # نقاط الدخول عبر Typer
├── tests/                # اختبارات Pytest
└── docs/                 # التوثيق
```

## التطوير

1. استنساخ المستودع
2. ثبت الاعتماديات بواسطة Poetry: `poetry install`
3. شغّل الاختبارات: `poetry run pytest`

## الترخيص

رخصة MIT — راجع ملف [LICENSE](LICENSE) للتفاصيل.
