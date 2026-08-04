# Candidate Skill-Matching Dashboard

داشبورد Streamlit لصاحب الشغل (Recruiter) يختار سكيلز معينة، فيطلعله كل
المرشحين المتوافقين معاها من الـ JSON اللي بيطلعها نموذج AI Resume
Extraction، مرتبين حسب نسبة التطابق. فيه كمان حاسبة ROI بتسحب أرقام حقيقية
(زمن استنتاج + دقة) من نتيجة تقييم الموديل لو موجودة.

## بنية المشروع

```
candidate-matcher-dashboard/
├── candidate_matcher_app.py     ← الملف الرئيسي (شغّله بـ streamlit)
├── requirements.txt             ← المكتبات المطلوبة
├── .gitignore
├── README.md
└── data/
    └── outputs/
        ├── sample_candidate_1.json   ← بيانات تجريبية (Demo) بس
        ├── sample_candidate_2.json
        └── sample_candidate_3.json
```

> ⚠️ **مهم:** ملفات `sample_candidate_*.json` دي بيانات وهمية (Fake) للعرض
> بس. لما تشغّل الموديل الحقيقي بتاعك (`ResumeExtractionEngine` من النوت
> بوك) وتحفظ نواتجه في `data/outputs/*.json`، الملف `.gitignore` هيمنعها
> من إنها تترفع على GitHub تلقائيًا — عشان دي بيانات حقيقية لأشخاص (خصوصية).

## 1) التشغيل محليًا

```bash
pip install -r requirements.txt
streamlit run candidate_matcher_app.py
```

هيفتح المتصفح على `http://localhost:8501` وتلاقي الـ 3 مرشحين التجريبيين
ظاهرين تلقائيًا.

## 2) رفعه على GitHub

```bash
cd candidate-matcher-dashboard
git init
git add .
git commit -m "Initial commit: candidate skill-matching dashboard"
git branch -M main
git remote add origin https://github.com/<username>/<repo-name>.git
git push -u origin main
```

## 3) نشره أونلاين (اختياري) عبر Streamlit Community Cloud

1. روح على [share.streamlit.io](https://share.streamlit.io) وسجّل دخول بحساب GitHub.
2. اختار "New app" → حدد الريبو ده → الفرع `main` → الملف `candidate_matcher_app.py`.
3. اضغط Deploy — هيقرأ `requirements.txt` تلقائيًا ويثبت المكتبات، ويطلعلك رابط عام تقدر تحطه في تسليم المشروع أو الـ README بتاعك.

## 4) توصيل بيانات حقيقية

بعد ما تشغّل `ResumeExtractionEngine` على سير ذاتية حقيقية، احفظ كل نتيجة
كملف JSON مستقل جوه `data/outputs/` (نفس الفورمات الموجود في ملفات الـ
sample). الداشبورد بيقرأ أي ملف `.json` في المجلد ده تلقائيًا من غير أي
كود إضافي.

لو عايز تربط حاسبة الـ ROI بأرقام حقيقية كمان، شغّل `full_evaluation.py`
(من مرحلة التقييم في النوت بوك) لينتج `data/outputs/evaluation_report.json`
— الداشبورد هيكتشفه ويستخدمه تلقائيًا بدل القيم الافتراضية.

