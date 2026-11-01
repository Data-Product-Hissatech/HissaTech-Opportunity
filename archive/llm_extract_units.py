"""
يستخدم Claude (Haiku) لتحليل وصف كل إعلان بيع:
1. يستخرج الوحدات المتعددة لو الإعلان مشروع (يطلّع صف منفصل لكل وحدة)
2. يتحقق من تطابق المساحة/العمر/الغرف/الحمامات بالوصف مع الأعمدة عندنا
3. يصحّح القيم الغلط تلقائيًا بالقيمة الصحيحة من الوصف

الاستخدام:
    python llm_extract_units.py                 # يشتغل على كل الملف
    python llm_extract_units.py --test          # عيّنة تجريبية: 10 مشاريع محتملة + 10 عاديين
"""

import pandas as pd
import os
import json
import re
import sys
import time
import urllib.request

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
INPUT_PATH = os.path.join(DATA_DIR, "final_combined_report.csv")
OUTPUT_PATH = os.path.join(DATA_DIR, "llm_verified_units.csv")

API_URL = "https://api.anthropic.com/v1/messages"
MODEL = "claude-haiku-4-5-20251001"

# مؤشرات تدل إن الإعلان يحتمل يكون مشروع متعدد الوحدات
# (مبنية على فحص فعلي لأوصاف حقيقية بالبيانات، مو تخمين)
MULTI_UNIT_HINTS = [
    # صيغ الأسعار/المساحات المتفاوتة
    "يبدأ من", "تبدأ من", "أسعار تبدأ", "اسعار تبدأ",
    "أسعار مختلفة", "اسعار مختلفة", "بأسعار مختلفة", "باسعار مختلفة",
    "عروض مختلفة", "مساحات تبدأ", "مساحات متنوعة", "مساحات مختلفة", "بمساحات",
    "اسعار ومساحات", "أسعار ومساحات",
    # صيغ تعدد الوحدات صراحة
    "شقق متنوعة", "عدة وحدات", "وحدات متعددة", "يوجد شقق", "وهناك شقق",
    "عدد الشقق", "شقة رقم", "الشقة رقم",
    # صيغ المشاريع والأدوار
    "نماذج", "النموذج", "مشروع", "المشروع",
    "الدور الأول", "الدور الثاني", "أدوار", "الأدوار",
    "كل دور", "دور بصك", "خيارات",
]

PROMPT_TEMPLATE = """أنت محلل عقاري. حلّل وصف الإعلان التالي واستخرج المعلومات بدقة.

بيانات الإعلان المسجّلة عندنا:
- الحي: {district}
- المساحة: {area} م²
- عدد الغرف: {rooms} (ملاحظة: هذا الرقم يشمل غرف النوم + المجلس/الصالة المنفصلة مجتمعين، لو مذكور مجلس منفصل بالوصف اعتبره ضمن عدد الغرف)
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
    {{
      "unit_label": "وصف مختصر للوحدة (مثل: نموذج أ - الدور الأول)",
      "area_sqm": رقم أو null,
      "rooms": رقم أو null,
      "bathrooms": رقم أو null,
      "price": رقم أو null
    }}
  ],
  "data_conflicts": {{
    "district": اسم الحي الصحيح من نص الوصف (بالضبط زي ما يذكره النص) أو null لو مطابق/غير مذكور -- هذا فحص مهم جدًا، تأكد منه بعناية حتى لو باقي الحقول مطابقة,
    "area_sqm": القيمة الصحيحة من الوصف أو null لو مطابقة/غير مذكورة,
    "rooms": نفس الشي (تذكّر: عدد الغرف = غرف النوم + المجلس/الصالة المنفصلة مجتمعين -- لا تعتبره تعارض لو فرق العدد يفسّره وجود مجلس منفصل بالوصف),
    "bathrooms": نفس الشي,
    "age_years": نفس الشي
  }},
  "notes": "ملاحظة مختصرة جدًا لو فيه شي مهم، أو نص فارغ"
}}

قواعد مهمة:
- لو الإعلان لوحدة واحدة فقط: is_multi_unit = false، و units تحتوي عنصر واحد بمعلومات الوحدة.
- لا تخترع وحدات غير مذكورة صراحة بالوصف.
- بـ data_conflicts: ضع قيمة فقط لو الوصف يذكر رقمًا **مختلفًا** عن المسجّل عندنا. لو مطابق أو غير مذكور، ضع null.
- كل الأرقام يجب أن تكون أرقامًا فقط (بدون نص أو وحدات)."""


def call_claude(prompt, api_key):
    payload = json.dumps({
        "model": MODEL,
        "max_tokens": 1500,
        "messages": [{"role": "user", "content": prompt}],
    }).encode("utf-8")

    req = urllib.request.Request(
        API_URL, data=payload, method="POST",
        headers={
            "content-type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return "".join(block.get("text", "") for block in data.get("content", []))


def parse_json_response(text):
    """ينظّف الرد ويحوّله JSON (يشيل علامات markdown لو موجودة)"""
    cleaned = re.sub(r"```(?:json)?|```", "", text).strip()
    return json.loads(cleaned)


def pick_test_sample(df):
    """يختار 10 يحتمل إنهم مشاريع متعددة + 10 عاديين -- عشان نختبر الاتجاهين"""
    desc = df["description"].fillna("").astype(str)
    hint_mask = desc.apply(lambda d: any(h in d for h in MULTI_UNIT_HINTS))

    likely_multi = df[hint_mask].head(10)
    likely_single = df[~hint_mask].head(10)
    print(f"عيّنة تجريبية: {len(likely_multi)} يحتمل مشاريع + {len(likely_single)} عاديين")
    return pd.concat([likely_multi, likely_single], ignore_index=True)


def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("تحذير: ما لقيت ANTHROPIC_API_KEY بمتغيرات البيئة")
        return

    if not os.path.exists(INPUT_PATH):
        print(f"تحذير: ما لقيت {INPUT_PATH}")
        return

    df = pd.read_csv(INPUT_PATH, encoding="utf-8-sig")
    print(f"عدد الصفوف بالملف: {len(df)}")

    test_mode = "--test" in sys.argv
    if test_mode:
        df = pick_test_sample(df)

    print(f"سنعالج: {len(df)} عقار\n")

    output_rows = []
    failed = 0

    for i, (_, row) in enumerate(df.iterrows(), start=1):
        prompt = PROMPT_TEMPLATE.format(
            district=row.get("district"), area=row.get("area_sqm"), rooms=row.get("rooms"),
            bathrooms=row.get("bathrooms"), age=row.get("age_years"),
            price=row.get("price"), description=str(row.get("description", ""))[:4000],
        )

        try:
            response_text = call_claude(prompt, api_key)
            result = parse_json_response(response_text)
        except Exception as e:
            print(f"[{i}/{len(df)}] فشل تحليل {row['listing_id']}: {e}")
            failed += 1
            continue

        conflicts = result.get("data_conflicts", {}) or {}
        units = result.get("units", []) or [{}]

        for unit in units:
            new_row = row.to_dict()
            new_row["is_multi_unit"] = result.get("is_multi_unit", False)
            new_row["unit_label"] = unit.get("unit_label")
            # بيانات الوحدة تفوز لو موجودة، وإلا نرجع للتصحيح، وإلا القيمة الأصلية
            for field in ["area_sqm", "rooms", "bathrooms", "price"]:
                unit_val = unit.get(field)
                conflict_val = conflicts.get(field)
                if unit_val is not None:
                    new_row[field] = unit_val
                elif conflict_val is not None:
                    new_row[field] = conflict_val
            if conflicts.get("age_years") is not None:
                new_row["age_years"] = conflicts["age_years"]

            new_row["llm_corrections"] = json.dumps(
                {k: v for k, v in conflicts.items() if v is not None}, ensure_ascii=False
            )
            new_row["llm_notes"] = result.get("notes", "")

            # لو فيه تعارض بيانات حقيقي، نضيفه لعمود risks الموجود (مو نستبدله)
            found_conflicts = {k: v for k, v in conflicts.items() if v is not None}
            if found_conflicts:
                conflict_text = "تعارض بيانات اكتشفه التحليل الآلي: " + ", ".join(
                    f"{field}={value} بالوصف (مسجّل عندنا {row.get(field)})"
                    for field, value in found_conflicts.items()
                )
                existing_risks_raw = new_row.get("risks")
                existing_risks = "" if pd.isna(existing_risks_raw) else str(existing_risks_raw).strip()
                new_row["risks"] = (existing_risks + " | " + conflict_text) if existing_risks else conflict_text
            output_rows.append(new_row)

        if i % 10 == 0:
            print(f"[{i}/{len(df)}] تمت المعالجة...")
        time.sleep(0.3)  # تهدئة بسيطة بين الطلبات

    result_df = pd.DataFrame(output_rows)
    result_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print(f"\n--- الملخص ---")
    print(f"عقارات دخلت التحليل: {len(df)}")
    print(f"صفوف ناتجة (بعد تفكيك الوحدات المتعددة): {len(result_df)}")
    print(f"فشل تحليلها: {failed}")
    if len(result_df):
        multi_count = result_df["is_multi_unit"].sum()
        print(f"صفوف من إعلانات متعددة الوحدات: {multi_count}")
        corrected = (result_df["llm_corrections"] != "{}").sum()
        print(f"صفوف فيها تصحيح بيانات: {corrected}")
    print(f"\nتم الحفظ: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
