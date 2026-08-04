"""
Candidate Skill-Matching Dashboard
==================================
داشبورد Streamlit لصاحب الشغل (Recruiter/Employer) يسمح له باختيار
مجموعة سكيلز معينة، فيطلع له كل السير الذاتية (Candidates) اللي اتستخرجت
بواسطة نموذج AI Resume Intelligence Engine ومتوافقة مع السكيلز دي،
مرتبة حسب نسبة التطابق.

طريقة التشغيل:
    streamlit run candidate_matcher_app.py

المصدر الافتراضي للبيانات: مجلد فيه ملفات JSON، كل ملف بيمثل مرشح واحد
(نفس الفورمات اللي بيطلعه ResumeExtractionEngine في النوت بوك - data/outputs).
لو معندكش ملفات جاهزة، الأداة فيها زرار لرفع ملفات JSON مباشرة من الجهاز.
"""

import os
import re
import json
import glob
import tempfile
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# -----------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------
st.set_page_config(
    page_title="Candidate Skill Matcher",
    page_icon="🧑‍💻",
    layout="wide",
)

DEFAULT_CANDIDATES_DIR = "data/outputs"  # نفس المجلد اللي بيحفظ فيه inference_engine
MODEL_PATH = "resume_parser_final_model"  # مسار الموديل اللي عملتله fine-tuning وحفظته


# -----------------------------------------------------------------------
# تحميل الموديل (مرة واحدة بس، ومتخزن في الكاش عشان مايتحملش من جديد كل مرة)
# -----------------------------------------------------------------------
@st.cache_resource(show_spinner="⏳ جاري تحميل الموديل... (بيحصل مرة واحدة بس)")
def load_model(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return tokenizer, model


# -----------------------------------------------------------------------
# استخراج نص السيرة الذاتية من الملف المرفوع (PDF / DOCX / TXT)
# -----------------------------------------------------------------------
def extract_resume_text(uploaded_file) -> str:
    """بيرجع نص السيرة الذاتية كامل بغض النظر عن الصيغة."""
    suffix = os.path.splitext(uploaded_file.name)[1].lower()

    if suffix == ".txt":
        return uploaded_file.read().decode("utf-8", errors="ignore")

    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError:
            st.error("محتاج تتثبت مكتبة pypdf: pip install pypdf")
            return ""
        reader = PdfReader(uploaded_file)
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    if suffix == ".docx":
        try:
            import docx2txt
        except ImportError:
            st.error("محتاج تتثبت مكتبة docx2txt: pip install docx2txt")
            return ""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        text = docx2txt.process(tmp_path)
        os.unlink(tmp_path)
        return text or ""

    st.warning(f"صيغة الملف {suffix} غير مدعومة. استخدم PDF أو DOCX أو TXT.")
    return ""


# -----------------------------------------------------------------------
# بناء البرومبت وتشغيل الموديل على نص السيرة الذاتية عشان يطلع JSON منظم
#
# ⚠️ مهم جدًا: شكل الـ prompt هنا لازم يكون طبق الأصل من الفورمات اللي
# الموديل شاف وقت التدريب (fine-tuning)، وإلا الموديل هيديك نتايج ضعيفة
# أو غير منطقية لأنه هيتعامل مع البرومبت كأنه حاجة جديدة عليه.
#
# من الـ training cell اللي بعتهولي، شكل الـ INPUT كان بادئ بكلمة "Resume:"
# ثم النص كامل من غير أي تعديل (حتى لو فيه رموز encoding زي Â أو â€‹).
# لسه ناقص شكل الـ OUTPUT/label بالظبط (الجزء اللي بعد "Resume:")
# — لما تبعتهولي، حط الفورمات بتاعه في PROMPT_TEMPLATE تحت.
# -----------------------------------------------------------------------

# ✏️ عدّل السطر ده لما تعرف شكل الـ output/label المطلوب (مثال: "Category:" أو "JSON:")
OUTPUT_MARKER = "JSON:"  # TODO: استبدلها بنفس الكلمة اللي كانت مستخدمة وقت التدريب

PROMPT_TEMPLATE = "Resume:\n{resume_text}\n\n" + OUTPUT_MARKER + "\n"


def build_prompt(tokenizer, resume_text: str) -> str:
    """
    بيبني نفس شكل البرومبت اللي الموديل اتدرب عليه:
    "Resume:\n<النص>\n\n<OUTPUT_MARKER>\n"

    لو اتضح إن الموديل اتدرب بـ chat template (system/user roles) بدل
    النص الخام ده، استبدل الفنكشن دي باللي كانت موجودة قبل كده (تحت في التعليق).
    """
    return PROMPT_TEMPLATE.format(resume_text=resume_text)


# --- نسخة بديلة لو الموديل فعلاً اتدرب بفورمات chat template (system/user) ---
# def build_prompt(tokenizer, resume_text: str) -> str:
#     messages = [
#         {"role": "system", "content": "استخرج بيانات السيرة الذاتية في شكل JSON."},
#         {"role": "user", "content": f"Resume:\n{resume_text}"},
#     ]
#     return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def extract_json_block(text: str) -> Optional[str]:
    """بياخد أول { لحد آخر } في النص، عشان يتجاهل أي كلام زيادة الموديل طلّعه."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return match.group(0) if match else None


def run_resume_extraction(tokenizer, model, resume_text: str, max_new_tokens: int = 512) -> Dict[str, Any]:
    prompt = build_prompt(tokenizer, resume_text)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    raw_output = tokenizer.decode(generated, skip_special_tokens=True)

    json_block = extract_json_block(raw_output)
    if not json_block:
        raise ValueError(f"الموديل مرجعش JSON صالح. الناتج الخام:\n{raw_output}")

    return json.loads(json_block)


# -----------------------------------------------------------------------
# Helpers: Loading & Normalizing candidate JSON records
# -----------------------------------------------------------------------
def _extract_skill_names(skills_field: Any) -> List[str]:
    """
    السكيلز في الـ JSON ممكن تيجي بأشكال مختلفة حسب النموذج:
    - ["Python", "SQL", "AWS"]
    - [{"name": "Python", "level": "Advanced"}, ...]
    - "Python, SQL, AWS"  (نص واحد)
    الدالة دي بتوحد كل الأشكال دي لقائمة نصوص نضيفة.
    """
    names: List[str] = []

    if skills_field is None:
        return names

    if isinstance(skills_field, str):
        # فصل بالفاصلة لو جاءت كنص واحد
        parts = [p.strip() for p in skills_field.split(",")]
        return [p for p in parts if p]

    if isinstance(skills_field, dict):
        # أحياناً بتيجي مقسمة لفئات: {"technical": [...], "soft": [...]}
        for v in skills_field.values():
            names.extend(_extract_skill_names(v))
        return names

    if isinstance(skills_field, list):
        for item in skills_field:
            if isinstance(item, str):
                names.append(item.strip())
            elif isinstance(item, dict):
                candidate_name = (
                    item.get("name")
                    or item.get("skill")
                    or item.get("title")
                )
                if candidate_name:
                    names.append(str(candidate_name).strip())
        return names

    return names


def normalize_candidate(raw: Dict[str, Any], source_name: str) -> Dict[str, Any]:
    """توحيد شكل ملف JSON الواحد (مهما اختلفت أسماء المفاتيح) لبروفايل موحّد."""
    name = (
        raw.get("name")
        or raw.get("candidate_name")
        or raw.get("full_name")
        or source_name
    )
    email = raw.get("email") or raw.get("contact", {}).get("email", "") if isinstance(raw.get("contact"), dict) else raw.get("email", "")
    phone = raw.get("phone") or (raw.get("contact", {}).get("phone", "") if isinstance(raw.get("contact"), dict) else "")
    seniority = raw.get("seniority") or raw.get("experience_level") or raw.get("level") or "Not specified"

    skills_field = raw.get("skills") or raw.get("technical_skills") or raw.get("key_skills")
    skills = _extract_skill_names(skills_field)

    experience = raw.get("experience") or raw.get("work_experience") or raw.get("employment_history") or []
    education = raw.get("education") or raw.get("education_summary") or []
    certifications = raw.get("certifications") or []

    return {
        "source_file": source_name,
        "name": name,
        "email": email or "N/A",
        "phone": phone or "N/A",
        "seniority": seniority,
        "skills": skills,
        "skills_lower": {s.lower() for s in skills},
        "experience": experience,
        "education": education,
        "certifications": certifications,
        "raw": raw,
    }


@st.cache_data(show_spinner=False)
def load_candidates_from_dir(directory: str) -> List[Dict[str, Any]]:
    candidates = []
    if not os.path.isdir(directory):
        return candidates
    for filepath in sorted(glob.glob(os.path.join(directory, "*.json"))):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                raw = json.load(f)
            candidates.append(normalize_candidate(raw, os.path.basename(filepath)))
        except (json.JSONDecodeError, OSError):
            continue
    return candidates


def load_candidates_from_uploads(uploaded_files) -> List[Dict[str, Any]]:
    candidates = []
    for uf in uploaded_files:
        try:
            raw = json.load(uf)
            candidates.append(normalize_candidate(raw, uf.name))
        except json.JSONDecodeError:
            st.warning(f"⚠️ الملف {uf.name} مش JSON صالح، تم تجاهله.")
    return candidates


# -----------------------------------------------------------------------
# تخزين المرشحين اللي اتضافوا عن طريق الموديل مباشرة (تفضل موجودة طول ما الجلسة شغالة)
# -----------------------------------------------------------------------
if "model_candidates" not in st.session_state:
    st.session_state.model_candidates = []  # كل عنصر = candidate بعد normalize_candidate

# -----------------------------------------------------------------------
# Sidebar: Data source
# -----------------------------------------------------------------------
st.sidebar.title("⚙️ إعدادات مصدر البيانات")

data_source = st.sidebar.radio(
    "من فين هنجيب بيانات المرشحين؟",
    [
        "🤖 معالجة سيرة ذاتية جديدة بالموديل",
        "مجلد على السيرفر (data/outputs)",
        "رفع ملفات JSON جاهزة",
    ],
)

file_candidates: List[Dict[str, Any]] = []

if data_source == "مجلد على السيرفر (data/outputs)":
    candidates_dir = st.sidebar.text_input("مسار المجلد", value=DEFAULT_CANDIDATES_DIR)
    file_candidates = load_candidates_from_dir(candidates_dir)
    if not file_candidates:
        st.sidebar.warning("لم يتم العثور على ملفات JSON في هذا المسار.")
elif data_source == "رفع ملفات JSON جاهزة":
    uploaded = st.sidebar.file_uploader(
        "ارفع ملفات JSON (ملف واحد لكل مرشح)", type="json", accept_multiple_files=True
    )
    if uploaded:
        file_candidates = load_candidates_from_uploads(uploaded)

# كل المرشحين = اللي جايين من ملفات + اللي اتضافوا عن طريق الموديل في الجلسة الحالية
candidates: List[Dict[str, Any]] = file_candidates + st.session_state.model_candidates

st.sidebar.markdown("---")
st.sidebar.caption(f"عدد المرشحين المحمّلين: **{len(candidates)}**")
if st.session_state.model_candidates:
    st.sidebar.caption(f"منهم بالموديل مباشرة: **{len(st.session_state.model_candidates)}**")

# -----------------------------------------------------------------------
# Main Page
# -----------------------------------------------------------------------
st.title("🧑‍💻 Candidate Skill-Matching Dashboard")
st.caption("داشبورد لصاحب الشغل: اختار السكيلز المطلوبة وشوف أفضل المرشحين المتاحين فورًا.")

# =========================================================================
# قسم 1: معالجة سيرة ذاتية جديدة مباشرة بالموديل
# =========================================================================
if data_source == "🤖 معالجة سيرة ذاتية جديدة بالموديل":
    st.header("📄 خطوة 1: ارفع السيرة الذاتية عشان الموديل يحللها")

    resume_input_mode = st.radio(
        "طريقة إدخال السيرة الذاتية", ["رفع ملف (PDF / DOCX / TXT)", "لصق النص مباشرة"], horizontal=True
    )

    resume_text = ""
    resume_file_name = "resume_pasted.json"

    if resume_input_mode == "رفع ملف (PDF / DOCX / TXT)":
        resume_file = st.file_uploader("ارفع ملف السيرة الذاتية", type=["pdf", "docx", "txt"])
        if resume_file is not None:
            resume_text = extract_resume_text(resume_file)
            resume_file_name = resume_file.name
    else:
        resume_text = st.text_area("الصق نص السيرة الذاتية هنا", height=250)

    if resume_text:
        with st.expander("👀 معاينة النص المستخرج قبل التحليل"):
            st.text(resume_text[:3000])

    run_button = st.button("🔍 شغّل الموديل واستخرج البيانات", type="primary", disabled=not resume_text)

    if run_button and resume_text:
        try:
            tokenizer, model = load_model(MODEL_PATH)
            with st.spinner("🧠 الموديل بيحلل السيرة الذاتية..."):
                parsed_json = run_resume_extraction(tokenizer, model, resume_text)
            st.session_state["_last_parsed"] = parsed_json
            st.session_state["_last_parsed_source"] = resume_file_name
            st.success("✅ تم استخراج البيانات بنجاح! راجعها تحت وأضفها للقائمة.")
        except Exception as e:
            st.error(f"⚠️ حصل خطأ أثناء تشغيل الموديل: {e}")

    if st.session_state.get("_last_parsed"):
        st.subheader("📋 خطوة 2: راجع البيانات (تقدر تعدّل قبل الإضافة)")
        edited_json_str = st.text_area(
            "بيانات المرشح (JSON) - تقدر تصلّح أي حاجة هنا",
            value=json.dumps(st.session_state["_last_parsed"], ensure_ascii=False, indent=2),
            height=300,
        )

        if st.button("➕ أضف المرشح ده لقائمة المطابقة"):
            try:
                parsed = json.loads(edited_json_str)
                new_candidate = normalize_candidate(parsed, st.session_state["_last_parsed_source"])
                st.session_state.model_candidates.append(new_candidate)
                del st.session_state["_last_parsed"]
                st.success(f"تمت إضافة {new_candidate['name']} لقائمة المرشحين. اختار مصدر بيانات تاني من الشمال عشان تشوف المطابقة، أو كمّل ضيف سير ذاتية زيادة.")
                st.rerun()
            except json.JSONDecodeError:
                st.error("الـ JSON اللي في الصندوق مش صالح، راجع الصياغة.")

    st.markdown("---")
    st.info(
        "💡 بعد ما تضيف سير ذاتية بالموديل، اختار "
        "**'مجلد على السيرفر'** أو خلي القائمة الجانبية زي ما هي "
        "وانزل تحت في قسم 'اختار السكيلز' عشان تشوف كل المرشحين (بما فيهم اللي ضفتهم دلوقتي) ومطابقتهم."
    )

if not candidates:
    st.info("لا يوجد مرشحون محمّلون بعد. حلل سيرة ذاتية بالموديل، أو ارفع ملفات JSON، أو حدد مجلد صحيح من القائمة الجانبية.")
    st.stop()

# ابنِ قائمة موحّدة بكل السكيلز الموجودة عند كل المرشحين
all_skills = sorted({skill for c in candidates for skill in c["skills"] if skill})

col_filters, col_results = st.columns([1, 2.2])

with col_filters:
    st.subheader("🎯 اختار السكيلز المطلوبة")
    if not all_skills:
        st.warning("لم يتم العثور على سكيلز داخل ملفات المرشحين.")
        selected_skills: List[str] = []
    else:
        selected_skills = st.multiselect(
            "السكيلز (يمكن اختيار أكثر من واحدة)",
            options=all_skills,
            help="هنرشح المرشحين اللي عندهم أي (أو كل) السكيلز دي حسب وضع البحث تحت.",
        )

    match_mode = st.radio(
        "طريقة المطابقة",
        ["يحتوي على أي سكيل من المختارة (OR)", "لازم يمتلك كل السكيلز المختارة (AND)"],
        index=0,
    )

    min_match_pct = st.slider(
        "أقل نسبة تطابق مقبولة (%)", min_value=0, max_value=100, value=0, step=5
    )

    seniority_options = sorted({c["seniority"] for c in candidates if c["seniority"]})
    seniority_filter = st.multiselect("فلترة حسب الخبرة/المستوى (اختياري)", seniority_options)

with col_results:
    st.subheader("📋 نتائج المطابقة")

    if not selected_skills:
        st.info("اختار سكيل واحد على الأقل من القائمة على اليسار عشان تظهر النتائج.")
    else:
        selected_lower = {s.lower() for s in selected_skills}
        results = []

        for c in candidates:
            matched = c["skills_lower"] & selected_lower
            if not matched:
                continue

            if match_mode.startswith("لازم") and not selected_lower.issubset(c["skills_lower"]):
                continue

            match_pct = round((len(matched) / len(selected_lower)) * 100, 1)
            if match_pct < min_match_pct:
                continue

            if seniority_filter and c["seniority"] not in seniority_filter:
                continue

            results.append(
                {
                    "الاسم": c["name"],
                    "نسبة التطابق %": match_pct,
                    "السكيلز المتطابقة": ", ".join(sorted(matched)),
                    "كل السكيلز": ", ".join(c["skills"]),
                    "المستوى": c["seniority"],
                    "البريد": c["email"],
                    "الهاتف": c["phone"],
                    "الملف": c["source_file"],
                }
            )

        results.sort(key=lambda r: r["نسبة التطابق %"], reverse=True)

        if not results:
            st.warning("لا يوجد مرشحون مطابقون لهذه المعايير.")
        else:
            st.success(f"تم العثور على {len(results)} مرشح مطابق.")
            df_results = pd.DataFrame(results)
            st.dataframe(df_results, use_container_width=True, hide_index=True)

            st.markdown("---")
            st.subheader("🔍 تفاصيل المرشح")
            chosen_name = st.selectbox(
                "اختار مرشح لعرض تفاصيله الكاملة", df_results["الاسم"].tolist()
            )
            chosen_candidate = next(
                (c for c in candidates if c["name"] == chosen_name), None
            )
            if chosen_candidate:
                with st.expander(f"البروفايل الكامل - {chosen_candidate['name']}", expanded=True):
                    st.write(f"**البريد الإلكتروني:** {chosen_candidate['email']}")
                    st.write(f"**الهاتف:** {chosen_candidate['phone']}")
                    st.write(f"**المستوى:** {chosen_candidate['seniority']}")
                    st.write("**السكيلز:**", ", ".join(chosen_candidate["skills"]) or "غير محدد")

                    st.write("**الخبرات العملية:**")
                    exp = chosen_candidate["experience"]
                    if isinstance(exp, list) and exp:
                        for job in exp:
                            if isinstance(job, dict):
                                role = job.get("title", job.get("role", "Position"))
                                company = job.get("company", job.get("employer", "Company"))
                                st.write(f"- {role} @ {company}")
                            else:
                                st.write(f"- {job}")
                    else:
                        st.write("لا يوجد بيانات خبرة.")

                    st.write("**التعليم:**")
                    edu = chosen_candidate["education"]
                    if isinstance(edu, list) and edu:
                        for e in edu:
                            st.write(f"- {e}")
                    else:
                        st.write("لا يوجد بيانات تعليم.")

                    certs = chosen_candidate["certifications"]
                    if certs:
                        st.write("**الشهادات:**")
                        for cert in certs:
                            st.write(f"- {cert}")

                    st.download_button(
                        "⬇️ تحميل بيانات المرشح (JSON)",
                        data=json.dumps(chosen_candidate["raw"], ensure_ascii=False, indent=2),
                        file_name=f"{chosen_candidate['name']}.json",
                        mime="application/json",
                    )

st.markdown("---")
st.caption(
    "💡 ملاحظة: الداشبورد ده بيقرأ ملفات JSON الناتجة من ResumeExtractionEngine "
    "(المشروع الأصلي بتاع الـ Fine-Tuned LLM). كل ما تشغّل الموديل على سير ذاتية جديدة "
    "وتتحفظ في مجلد data/outputs، هتظهر تلقائيًا هنا."
)

# =========================================================================
# قسم: حاسبة العائد على الاستثمار (ROI) — بأرقام حقيقية مش افتراضية
# =========================================================================
st.markdown("---")
st.header("💰 حاسبة العائد على الاستثمار (ROI) - بأرقام حقيقية")
st.caption(
    "القسم ده بيسحب أرقام حقيقية (زمن استنتاج فعلي + دقة استخراج فعلية) من ملف "
    "evaluation_report.json الناتج عن full_evaluation.py، بدل ما يعتمد على أرقام "
    "افتراضية زي الـ 4 ثواني/سيرة الموجودة في خلية الـ ROI الأصلية بالنوت بوك."
)

EVAL_REPORT_PATH = st.text_input(
    "مسار تقرير التقييم الحقيقي",
    value="data/outputs/evaluation_report.json",
    key="eval_report_path",
)

real_metrics = None
if os.path.exists(EVAL_REPORT_PATH):
    try:
        with open(EVAL_REPORT_PATH, "r", encoding="utf-8") as f:
            real_metrics = json.load(f)
    except json.JSONDecodeError:
        st.warning("⚠️ ملف تقرير التقييم موجود لكنه مش JSON صالح.")
else:
    st.info(
        "لم يتم العثور على ملف تقرير تقييم حقيقي. شغّل `full_evaluation.py` على الموديل "
        "بعد التدريب عشان تحصل على أرقام حقيقية (وقت استنتاج فعلي + دقة فعلية) بدل "
        "الافتراضات. هنستخدم دلوقتي قيمًا افتراضية للتوضيح فقط."
    )

col_a, col_b = st.columns(2)
with col_a:
    monthly_volume = st.number_input("عدد السير الذاتية الشهرية", min_value=1, value=10000, step=500)
    recruiter_hourly_rate = st.number_input("تكلفة ساعة الـ Recruiter ($)", min_value=1.0, value=50.0, step=5.0)
with col_b:
    manual_minutes = st.number_input("متوسط وقت الفرز اليدوي لكل سيرة (دقيقة)", min_value=0.5, value=6.0, step=0.5)
    gpu_hour_cost = st.number_input("تكلفة ساعة الـ GPU ($)", min_value=0.1, value=1.5, step=0.1)

if real_metrics and "avg_inference_time_seconds" in real_metrics:
    default_ai_secs = real_metrics["avg_inference_time_seconds"]
    st.success(
        f"✅ بيتم استخدام متوسط زمن استنتاج **حقيقي مقاس فعليًا** = {default_ai_secs} ثانية/سيرة "
        f"(من تقييم على {real_metrics.get('num_samples_evaluated', '?')} عينة test فعلية)."
    )
else:
    default_ai_secs = 4.0
    st.warning("⚠️ بيتم استخدام قيمة افتراضية (4 ثواني/سيرة) لأنه مفيش تقرير تقييم حقيقي متاح.")

ai_secs_per_resume = st.number_input(
    "زمن معالجة الـ AI لكل سيرة (ثانية)", min_value=0.1, value=float(default_ai_secs), step=0.1
)

# --- الحسبة الأساسية (استبدال كامل، زي خلية الـ ROI الأصلية في النوت بوك) ---
manual_hours = (monthly_volume * manual_minutes) / 60.0
manual_cost = manual_hours * recruiter_hourly_rate

ai_hours = (monthly_volume * ai_secs_per_resume) / 3600.0
ai_cost = max(50.0, ai_hours * gpu_hour_cost)

savings = manual_cost - ai_cost
reduction_pct = (savings / manual_cost) * 100 if manual_cost else 0

st.subheader("📈 النتيجة الأساسية (استبدال كامل - Simple)")
m1, m2, m3 = st.columns(3)
m1.metric("التكلفة اليدوية الشهرية", f"${manual_cost:,.2f}")
m2.metric("تكلفة الـ AI الشهرية", f"${ai_cost:,.2f}")
m3.metric("التوفير الشهري", f"${savings:,.2f}", f"{reduction_pct:.1f}%")

# --- نسخة واقعية (Hybrid): بتاخد في الاعتبار إن مش كل استخراج بيطلع صح 100% ---
st.subheader("🧪 نسخة واقعية (Hybrid Review)")
st.caption(
    "بتفترض إن نسبة من السير الذاتية (بمقدار نسبة الخطأ الحقيقية) هتحتاج مراجعة "
    "يدوية سريعة، بدل افتراض إن الـ AI بيستبدل الـ Recruiter بنسبة 100%."
)

if real_metrics and "overall_field_exact_match_accuracy" in real_metrics:
    default_accuracy = real_metrics["overall_field_exact_match_accuracy"]
    st.caption(f"📊 دقة الاستخراج الحقيقية المقاسة (Per-field Overall): {default_accuracy}%")
else:
    default_accuracy = 90.0
    st.caption("⚠️ لا يوجد رقم دقة حقيقي متاح، مستخدمين افتراض 90% للتوضيح فقط.")

field_accuracy = st.slider("دقة الاستخراج الكلية المستخدمة في الحساب (%)", 0.0, 100.0, float(default_accuracy), 0.5)
review_minutes = st.number_input(
    "وقت المراجعة السريعة للسيرة المحتاجة تدقيق يدوي (دقيقة)", min_value=0.5, value=2.0, step=0.5
)

error_rate = (100 - field_accuracy) / 100
flagged_resumes = monthly_volume * error_rate
review_hours = (flagged_resumes * review_minutes) / 60.0
review_cost = review_hours * recruiter_hourly_rate

hybrid_cost = ai_cost + review_cost
hybrid_savings = manual_cost - hybrid_cost
hybrid_reduction_pct = (hybrid_savings / manual_cost) * 100 if manual_cost else 0

h1, h2, h3 = st.columns(3)
h1.metric("سير محتاجة مراجعة يدوية", f"{flagged_resumes:,.0f} سيرة")
h2.metric("التكلفة الكلية (AI + مراجعة)", f"${hybrid_cost:,.2f}")
h3.metric("التوفير الواقعي", f"${hybrid_savings:,.2f}", f"{hybrid_reduction_pct:.1f}%")

# --- عرض تفاصيل التقييم الحقيقي لو موجودة (Per-field + Confusion) ---
if real_metrics:
    with st.expander("📋 تفاصيل التقييم الحقيقي الكامل (Per-field breakdown + Confusion)"):
        st.write("**دقة كل حقل لوحده:**")
        if "per_field_accuracy" in real_metrics:
            pf_df = pd.DataFrame(
                list(real_metrics["per_field_accuracy"].items()), columns=["الحقل", "الدقة %"]
            ).sort_values("الدقة %", ascending=False)
            st.dataframe(pf_df, use_container_width=True, hide_index=True)
            st.bar_chart(pf_df.set_index("الحقل"))

        if "skills_confusion" in real_metrics:
            st.write("**دقة استخراج السكيلز (Multi-label Confusion):**")
            sc = real_metrics["skills_confusion"]
            st.write(
                f"TP={sc['TP']} | FP={sc['FP']} | FN={sc['FN']} | TN={sc['TN']}  →  "
                f"Precision={sc['precision']} | Recall={sc['recall']} | F1={sc['f1']}"
            )

        if "seniority_confusion_matrix_plot" in real_metrics and os.path.exists(
            real_metrics["seniority_confusion_matrix_plot"]
        ):
            st.write("**Confusion Matrix - Seniority:**")
            st.image(real_metrics["seniority_confusion_matrix_plot"])

        if "skills_confusion_matrix_plot" in real_metrics and os.path.exists(
            real_metrics["skills_confusion_matrix_plot"]
        ):
            st.write("**Confusion Matrix - Skills:**")
            st.image(real_metrics["skills_confusion_matrix_plot"])
