"""
يبني الجدول النهائي للعرض على المدير:
1. يختار ~20 عقار من yield_from_sakani_indicators.csv (عائد 7-9%، ثقة سكني عالية،
   موزّعين بالتساوي على المناطق -- عيّنة تمثيلية، مو أعلى 20 عشوائي)
2. يشغّل الـ LLM عليهم (استخراج وحدات متعددة + تحقق تعارض بيانات)
3. يضيف التوصية النهائية الموحدة ودرجة الثقة
4. يطلّع جدول نهائي بالأعمدة المطلوبة بالضبط، بدون رقم التواصل
"""

import pandas as pd
import os
import json
import re
import time
import urllib.request

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
INPUT_PATH = os.path.join(DATA_DIR, "yield_from_sakani_indicators.csv")
OUTPUT_PATH = os.path.join(DATA_DIR, "manager_presentation_report.csv")

API_URL = "https://api.anthropic.com/v1/messages"
MODEL = "claude-haiku-4-5-20251001"

YIELD_MIN, YIELD_MAX = 5.5, 15.0
TARGET_SAMPLE_SIZE = 50
MIN_STRONG_TARGET = 20   # أقل عدد "فرصة قوية" نضمنه بالعيّنة النهائية
STRONG_BUFFER = 8        # هامش أمان إضافي -- بعض المرشحين ممكن ينزلون بعد فحص LLM
                          # (لو الوصف فيه تعارض بيانات)، فنختار أكثر من 20 بالبداية
STRONG_COMPARISON_MIN_DEALS = 10

PROMPT_TEMPLATE = """أنت محلل عقاري. حلّل وصف الإعلان التالي واستخرج المعلومات بدقة.

بيانات الإعلان المسجّلة عندنا:
- نوع العقار المسجّل: شقة
- الحي: {district}
- المساحة: {area} م²
- عدد الغرف: {rooms}
- عدد الحمامات: {bathrooms}
- عمر العقار: {age} سنة
- السعر: {price} ريال

نص الوصف:
\"\"\"
{description}
\"\"\"

المطلوب: أرجع JSON فقط (بدون أي نص إضافي أو علامات markdown) بهذا الشكل بالضبط:

{{
  "is_multi_unit": true أو false,
  "units": [
    {{"unit_label": "وصف مختصر للوحدة", "area_sqm": رقم أو null, "rooms": رقم أو null,
      "bathrooms": رقم أو null, "price": رقم أو null}}
  ],
  "actual_property_type": صنّف نوع العقار الحقيقي من نص الوصف بدقة، اختر قيمة واحدة بالضبط من هذي القائمة: "شقة" أو "دور" أو "فيلا" أو "دوبلكس" أو "استوديو" أو "غير محدد" -- اعتمد على وصف العقار نفسه (مو عنوان الإعلان بس)، مثال: لو الوصف يقول "دور سكني للبيع" أو "الدور بالكامل" فالنوع "دور" حتى لو عنوان الإعلان يقول شقة,
  "data_conflicts": {{
    "district": اسم الحي الصحيح من نص الوصف (بالضبط زي ما يذكره النص) أو null لو مطابق/غير مذكور -- هذا فحص مهم جدًا، تأكد منه بعناية حتى لو باقي الحقول مطابقة,
    "area_sqm": القيمة الصحيحة من الوصف أو null,
    "rooms": نفس الشي, "bathrooms": نفس الشي, "age_years": نفس الشي
  }},
  "notes": "ملاحظة مختصرة جدًا أو نص فارغ"
}}

قواعد: لا تخترع وحدات غير مذكورة صراحة. بـ data_conflicts ضع قيمة فقط لو الوصف يذكر رقمًا مختلفًا عن المسجّل."""


def call_claude(prompt, api_key):
    payload = json.dumps({
        "model": MODEL, "max_tokens": 1500,
        "messages": [{"role": "user", "content": prompt}],
    }).encode("utf-8")
    req = urllib.request.Request(
        API_URL, data=payload, method="POST",
        headers={"content-type": "application/json", "x-api-key": api_key, "anthropic-version": "2023-06-01"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return "".join(block.get("text", "") for block in data.get("content", []))


def parse_json_response(text):
    cleaned = re.sub(r"```(?:json)?|```", "", text).strip()
    return json.loads(cleaned)


MAX_AGE_YEARS = 10  # نستبعد العقارات الأقدم من هذا من العيّنة كليًا

DUPLEX_PATTERN = re.compile(r"دوبلكس|دبلوكس|فلا\s*دوبلكس|فلة\s*دوبلكس")
# نتقبّل "ال" التعريف اختياريًا قبل "دور" وقبل الرقم الترتيبي (الوصف غالبًا
# يقول "الدور الثاني"، مو "دور ثاني")
FLOOR_PATTERN = re.compile(r"(ال)?دور\s*(ال)?(أول|ثاني|ثالث|رابع|اول|تاني|ارضي)")
# إعلان صريح لنوع العقار نفسه كـ"دور" (زي "دور سكني للبيع") -- إشارة قوية
# لحالها، تميّز حالة "العقار نفسه دور" عن حالة "شقة عادية بالدور الثاني"
EXPLICIT_FLOOR_TYPE = re.compile(r"دور\s+سكني")


def is_actually_floor_unit(description):
    """يكتشف لو العقار فعليًا 'دور' مستقل، مو شقة حقيقية -- إما دور ضمن فلا
    دوبلكس، أو معلَن صراحة كـ'دور سكني' بالوصف"""
    text = str(description or "")
    is_duplex_floor = bool(DUPLEX_PATTERN.search(text) and FLOOR_PATTERN.search(text))
    is_explicit_floor = bool(EXPLICIT_FLOOR_TYPE.search(text))
    return is_duplex_floor or is_explicit_floor


def pick_diverse_sample(df):
    """يختار عيّنة 50 عقار، تضمن MIN_STRONG_TARGET (20) على الأقل من الفرص
    القوية المحتملة (سعر عادل + عائد >=7%)، والباقي متنوع من المناطق"""
    before_floor_filter = len(df)
    df = df[~df["description"].apply(is_actually_floor_unit)].copy()
    print(f"استبعدنا {before_floor_filter - len(df)} عقار وصفه يوضح إنه 'دور' مو شقة فعلية")

    pool = df[
        (df["expected_yield_pct_sakani"] >= YIELD_MIN)
        & (df["expected_yield_pct_sakani"] <= YIELD_MAX)
        & (df["sakani_trusted"] == True)
        & (df["age_years"] <= MAX_AGE_YEARS)
    ].copy()
    print(f"عقارات بنطاق العائد {YIELD_MIN}-{YIELD_MAX}% وثقة سكني موثوقة: {len(pool)}")

    if len(pool) == 0:
        return pool

    # مرشحين "فرصة قوية محتملة" -- سعر عادل رسميًا + عائد قوي (>=7%)
    # نأخذ أكثر من MIN_STRONG_TARGET كهامش أمان (بعضهم ممكن ينزل بعد فحص LLM
    # لو الوصف فيه تعارض بيانات لم نكتشفه بعد بهالمرحلة)
    is_fair_price = pool["verdict_price"].astype(str).str.contains("عادل", na=False)
    is_strong_yield = pool["expected_yield_pct_sakani"] >= 7.0
    strong_pool = pool[is_fair_price & is_strong_yield].sort_values("expected_yield_pct_sakani", ascending=False)

    strong_take = min(len(strong_pool), MIN_STRONG_TARGET + STRONG_BUFFER)
    strong_sample = strong_pool.head(strong_take)
    print(f"مرشحين فرصة قوية محتملة (سعر عادل + عائد>=7%): {len(strong_pool)}، أخذنا {len(strong_sample)}")

    if len(strong_sample) < MIN_STRONG_TARGET:
        print(f"⚠️ تحذير: بس {len(strong_sample)} مرشح قوي متوفر، أقل من الهدف {MIN_STRONG_TARGET}")

    # الباقي (لحد 50) نوزّعه متنوع على المناطق من بقية العيّنة (مو الفرص القوية)
    remaining_slots = TARGET_SAMPLE_SIZE - len(strong_sample)
    rest_pool = pool[~pool["listing_id"].isin(strong_sample["listing_id"])]

    diverse_parts = []
    if remaining_slots > 0 and len(rest_pool) > 0:
        directions = rest_pool["direction"].dropna().unique()
        per_direction = max(1, remaining_slots // max(len(directions), 1))
        for d in directions:
            part = rest_pool[rest_pool["direction"] == d].sort_values("expected_yield_pct_sakani", ascending=False).head(per_direction)
            diverse_parts.append(part)

    diverse_sample = pd.concat(diverse_parts, ignore_index=True) if diverse_parts else pd.DataFrame()

    sample = pd.concat([strong_sample, diverse_sample], ignore_index=True)

    if len(sample) < TARGET_SAMPLE_SIZE:
        remaining = pool[~pool["listing_id"].isin(sample["listing_id"])]
        extra = remaining.sort_values("expected_yield_pct_sakani", ascending=False).head(TARGET_SAMPLE_SIZE - len(sample))
        sample = pd.concat([sample, extra], ignore_index=True)

    sample = sample.drop_duplicates(subset="listing_id").head(TARGET_SAMPLE_SIZE)
    print(f"العيّنة النهائية: {len(sample)} عقار، موزّعة على {sample['direction'].nunique()} مناطق")
    print(sample["direction"].value_counts().to_string())
    return sample


def has_conflicts(value):
    text = str(value or "").strip()
    return text not in ("", "{}", "nan", "None")


def has_severe_conflict(value):
    """يكتشف تحديدًا لو التعارض يشمل الحي أو نوع العقار -- أخطر من باقي الحقول،
    لأنه يبطل كل حسابات الإيجار والسعر المبنية على الحي/النوع المسجّل (احتُسبت
    قبل هالفحص)"""
    if not has_conflicts(value):
        return False
    try:
        parsed = json.loads(value)
        return "district" in parsed or "actual_property_type" in parsed
    except (json.JSONDecodeError, TypeError):
        return False


def severe_conflict_reason(value):
    """يحدد أي نوع من التعارض الخطير موجود، لصياغة رسالة دقيقة"""
    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return None
    has_district = "district" in parsed
    has_type = "actual_property_type" in parsed
    if has_district and has_type:
        return "الوصف يذكر حي ونوع عقار مختلفين عن المسجّل"
    if has_district:
        return "الوصف يذكر حي مختلف عن المسجّل"
    if has_type:
        return f"الوصف يوضح إن نوع العقار الحقيقي '{parsed['actual_property_type']}' مختلف عن المفترض (شقة)"
    return None


def compute_confidence(row):
    signals = 0
    if row.get("sakani_trusted") is True:
        signals += 1
    deals_count = row.get("comparable_sale_deals_count")
    if pd.notna(deals_count) and deals_count >= STRONG_COMPARISON_MIN_DEALS:
        signals += 1
    if has_severe_conflict(row.get("llm_corrections")):
        signals -= 10  # تعارض الحي أو النوع يُسقط الثقة لأدنى درجة دائمًا
    elif has_conflicts(row.get("llm_corrections")):
        signals -= 1
    if signals >= 2:
        return "عالية"
    elif signals == 1:
        return "متوسطة"
    return "منخفضة"


def compute_final_verdict(row):
    verdict_price = str(row.get("verdict_price") or "")
    conflicts = has_conflicts(row.get("llm_corrections"))

    if has_severe_conflict(row.get("llm_corrections")):
        reason = severe_conflict_reason(row.get("llm_corrections"))
        return "🔴 لا ينصح بها", f"{reason} -- كل حسابات الإيجار والسعر غير موثوقة، تحتاج مراجعة يدوية كاملة قبل أي قرار"
    if "REVIEW" in verdict_price:
        return "🔴 لا ينصح بها", "سعر البيع أرخص بشكل غير طبيعي مقارنة بصفقات رسمية مشابهة -- تحقق من الصك والمساحة أولًا"
    if not verdict_price or verdict_price == "nan":
        return "⚪ غير مكتمل", "ما توفرت صفقات رسمية كافية للتحقق من عدالة السعر"
    if "مبالغ" in verdict_price:
        return "🟡 تحتاج مراجعة", "سعر البيع أعلى من صفقات رسمية مشابهة -- تفاوض على السعر قبل أي قرار"
    if conflicts:
        return "🟡 تحتاج مراجعة", "السعر عادل، لكن فيه تعارض بين بيانات الإعلان والوصف -- تأكد من التفاصيل الفعلية"
    return "🟢 فرصة قوية", "عائد إيجاري موثّق رسميًا من سكني، وسعر بيع عادل مقارنة بصفقات وزارة العدل، وبدون أي تعارض بالبيانات"


def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("تحذير: ما لقيت ANTHROPIC_API_KEY بمتغيرات البيئة")
        return
    if not os.path.exists(INPUT_PATH):
        print(f"تحذير: ما لقيت {INPUT_PATH}")
        return

    df = pd.read_csv(INPUT_PATH, encoding="utf-8-sig")
    print(f"عدد الصفوف الكلي: {len(df)}")

    sample = pick_diverse_sample(df)
    if len(sample) == 0:
        print("ما لقينا أي عقار مطابق للشروط -- جرّب توسيع نطاق العائد")
        return

    output_rows = []
    for i, (_, row) in enumerate(sample.iterrows(), start=1):
        prompt = PROMPT_TEMPLATE.format(
            district=row.get("district"), area=row.get("area_sqm"), rooms=row.get("rooms"),
            bathrooms=row.get("bathrooms"), age=row.get("age_years"),
            price=row.get("price"), description=str(row.get("description", ""))[:4000],
        )
        try:
            response_text = call_claude(prompt, api_key)
            result = parse_json_response(response_text)
        except Exception as e:
            print(f"[{i}/{len(sample)}] فشل تحليل {row['listing_id']}: {e}")
            result = {"is_multi_unit": False, "units": [{}], "data_conflicts": {}, "notes": ""}

        conflicts = result.get("data_conflicts", {}) or {}
        units = result.get("units", []) or [{}]

        for unit in units:
            new_row = row.to_dict()
            new_row["is_multi_unit"] = result.get("is_multi_unit", False)
            new_row["unit_label"] = unit.get("unit_label")
            for field in ["area_sqm", "rooms", "bathrooms", "price"]:
                unit_val = unit.get(field)
                conflict_val = conflicts.get(field)
                if unit_val is not None:
                    new_row[field] = unit_val
                elif conflict_val is not None:
                    new_row[field] = conflict_val
            if conflicts.get("age_years") is not None:
                new_row["age_years"] = conflicts["age_years"]

            found_conflicts = {k: v for k, v in conflicts.items() if v is not None}
            new_row["llm_corrections"] = json.dumps(found_conflicts, ensure_ascii=False)
            new_row["llm_notes"] = result.get("notes", "")

            # نوع العقار الحقيقي من تحليل الـ LLM -- يستبدل القيمة الثابتة "شقة"
            actual_type = result.get("actual_property_type") or "شقة"
            new_row["نوع_العقار"] = actual_type
            type_mismatch = actual_type not in ("شقة", "غير محدد", None)
            if type_mismatch:
                type_conflict_text = f"نوع العقار الحقيقي '{actual_type}' مختلف عن المفترض 'شقة' -- كل حسابات الإيجار ومقارنة السعر اتبنت على افتراض شقة، غير موثوقة لهذا العقار"
                found_conflicts["actual_property_type"] = actual_type

            if found_conflicts:
                conflict_text = "تعارض بيانات اكتشفه التحليل الآلي: " + ", ".join(
                    f"{field}={value} بالوصف (مسجّل عندنا {row.get(field)})"
                    for field, value in found_conflicts.items() if field != "actual_property_type"
                )
                if type_mismatch:
                    conflict_text = (conflict_text + " | " + type_conflict_text) if conflict_text else type_conflict_text
                existing_risks_raw = new_row.get("risks")
                existing_risks = "" if pd.isna(existing_risks_raw) else str(existing_risks_raw).strip()
                new_row["risks"] = (existing_risks + " | " + conflict_text) if existing_risks else conflict_text
                new_row["llm_corrections"] = json.dumps(found_conflicts, ensure_ascii=False)

            output_rows.append(new_row)

        print(f"[{i}/{len(sample)}] تمت المعالجة -- {row.get('district')}")
        time.sleep(0.3)

    result_df = pd.DataFrame(output_rows)

    verdicts = result_df.apply(compute_final_verdict, axis=1)
    result_df["التوصية_النهائية"] = [v[0] for v in verdicts]
    result_df["سبب_التوصية_النهائية"] = [v[1] for v in verdicts]
    result_df["درجة_الثقة"] = result_df.apply(compute_confidence, axis=1)

    result_df = result_df.rename(columns={
        "expected_yield_pct_sakani": "yield_pct",
        "expected_annual_rent_sakani": "expected_annual_rent",
    })

    # عمود نوع العقار -- يُحدَّد لكل صف من تحليل الـ LLM بالحلقة أعلاه (مو ثابت "شقة"
    # بعد اليوم)، عمود احتياطي فقط لو صف ما مرّ على الحلقة لأي سبب
    if "نوع_العقار" not in result_df.columns:
        result_df["نوع_العقار"] = "شقة"

    final_cols = [
        "listing_id", "url", "title", "district", "city", "direction",
        "التوصية_النهائية", "درجة_الثقة", "سبب_التوصية_النهائية",
        "نوع_العقار", "price", "area_sqm", "rooms", "bathrooms", "livings", "age_years", "مؤثثة",
        "yield_pct", "expected_annual_rent", "sakani_deals_count",
        "ratio" if "ratio" in result_df.columns else "price_ratio",
        "verdict_price",
        "verdict_price_reason" if "verdict_price_reason" in result_df.columns else None,
        "strengths", "risks",
        "is_multi_unit", "unit_label", "llm_corrections", "llm_notes",
        "description", "images",
    ]
    final_cols = [c for c in final_cols if c and c in result_df.columns]

    # أعمدة نستبعدها كليًا (مو مفيدة بعرض المدير: بيانات تقنية/إدارية داخلية)
    EXCLUDE_COLS = {
        "images_count", "advertiser_name", "advertiser_company", "advertiser_type",
        "created_at", "published_at", "last_update", "views", "date_scraped",
        "published", "price_text", "price_was_missing", "price_per_sqm",
        "anomaly_score", "is_anomaly", "رقم_التواصل",
    }
    remaining = [c for c in result_df.columns if c not in final_cols and c not in EXCLUDE_COLS]
    result_df = result_df[final_cols + remaining]

    rank = {"🟢 فرصة قوية": 0, "🟡 تحتاج مراجعة": 1, "⚪ غير مكتمل": 2, "🔴 لا ينصح بها": 3}
    result_df["_rank"] = result_df["التوصية_النهائية"].map(rank).fillna(9)
    result_df = result_df.sort_values(["_rank", "yield_pct"], ascending=[True, False]).drop(columns=["_rank"])

    result_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print(f"\n--- توزيع التوصية النهائية ---")
    print(result_df["التوصية_النهائية"].value_counts().to_string())
    print(f"\n--- توزيع درجة الثقة ---")
    print(result_df["درجة_الثقة"].value_counts().to_string())
    print(f"\nتم الحفظ: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
