"""
استخراج العقارات المؤجّرة من الأنواع الثانية (مو شقق) -- فلل، أدوار، عمائر،
مكاتب، أراضي. يستخدم نفس دوال الاستخراج المُختبرة من rented_extracted_no_api.py
(بدون أي API، Regex بحت، صفر تكلفة).

يقرأ من كل ملفات الأنواع المتوفرة، يدمجها بملف واحد، مع عمود "نوع_العقار"
يوضح مصدر كل صف.
"""

import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from rented_extracted_no_api import (
    RENTED_HINTS, extract_annual_rent, sanity_check_rent,
    extract_key_features, normalize_for_duplicate_check, is_owner_direct,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUTPUT_PATH = os.path.join(DATA_DIR, "rented_other_types_no_api.csv")

# كل الأنواع اللي ممكن تكون مؤجّرة -- (اسم الملف المتوقع، التسمية العربية)
# نجرب أكثر من اسم محتمل لكل نوع (normal لو موجود، أو الأساسي)
PROPERTY_TYPES = [
    ("villa", "فيلا"),
    ("floor", "دور"),
    ("building", "عمارة"),
    ("office", "مكتب"),
    ("land", "أرض"),
]


def load_type_file(type_key):
    """يحاول يقرأ ملف النوع -- يفضّل نسخة _normal لو موجودة، وإلا الأساسية"""
    candidates = [
        os.path.join(DATA_DIR, f"listings_{type_key}_normal.csv"),
        os.path.join(DATA_DIR, f"listings_{type_key}.csv"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return pd.read_csv(path, encoding="utf-8-sig"), path
    return None, None


def process_type(df, type_label):
    """يشغّل نفس منطق استخراج الإيجار المُختبر على نوع عقار معيّن"""
    mask = df["description"].fillna("").astype(str).str.contains(RENTED_HINTS, na=False)
    candidates = df[mask].copy()
    if len(candidates) == 0:
        return candidates

    rents, reasons = [], []
    for _, row in candidates.iterrows():
        rent, reason = extract_annual_rent(row.get("description"), row.get("price"))
        rent = sanity_check_rent(rent, row.get("price"))
        rents.append(rent)
        reasons.append(reason)

    candidates["actual_annual_rent"] = rents
    candidates["yield_pct"] = candidates.apply(
        lambda r: round(r["actual_annual_rent"] / r["price"] * 100, 2)
        if pd.notna(r["actual_annual_rent"]) and r.get("price") else None,
        axis=1
    )
    candidates["key_features"] = candidates["description"].apply(extract_key_features)

    owner_flags, owner_reasons = [], []
    for _, row in candidates.iterrows():
        is_owner, reason = is_owner_direct(row)
        owner_flags.append("نعم" if is_owner else "لا")
        owner_reasons.append(reason)
    candidates["من_المالك_مباشرة"] = owner_flags
    candidates["owner_detection_reason"] = owner_reasons

    candidates["نوع_العقار"] = type_label
    return candidates


def main():
    all_results = []

    for type_key, type_label in PROPERTY_TYPES:
        df, path = load_type_file(type_key)
        if df is None:
            print(f"{type_label} ({type_key}): ما لقينا ملف، تخطّينا")
            continue

        print(f"{type_label} ({type_key}): {len(df)} إعلان من {path}")
        result = process_type(df, type_label)
        print(f"  → مؤجّرة مكتشفة: {len(result)}")
        if len(result) > 0:
            all_results.append(result)

    if not all_results:
        print("ما لقينا أي عقار مؤجّر بأي نوع -- توقف هنا")
        return

    combined = pd.concat(all_results, ignore_index=True)
    print(f"\nإجمالي المؤجّرة من كل الأنواع: {len(combined)}")

    # حذف التكرار (نفس الوصف بأي نوع/رقم إعلان)
    combined["_normalized_desc"] = combined["description"].apply(normalize_for_duplicate_check)
    before_dedup = len(combined)
    combined = combined.drop_duplicates(subset="_normalized_desc", keep="first")
    combined = combined.drop(columns=["_normalized_desc"])
    print(f"حذفنا {before_dedup - len(combined)} إعلان مكرر")

    # نفس ترتيب أولوية الشقق: عائد محسوب أول تنازليًا، بعدين العمر تصاعديًا
    combined["_has_yield"] = combined["yield_pct"].notna()
    combined = combined.sort_values(
        ["_has_yield", "yield_pct", "age_years"],
        ascending=[False, False, True],
        na_position="last"
    ).drop(columns=["_has_yield"])

    cols = [c for c in ["listing_id", "url", "title", "published_at", "نوع_العقار", "district", "direction",
                          "price", "area_sqm", "rooms", "bathrooms", "age_years",
                          "actual_annual_rent", "yield_pct", "key_features",
                          "من_المالك_مباشرة", "owner_detection_reason",
                          "description"] if c in combined.columns]
    combined = combined[cols]

    # تنسيق فواصل الآلاف والنسبة المئوية -- نفس أسلوب ملف الشقق
    for col in ["price", "actual_annual_rent"]:
        if col in combined.columns:
            combined[col] = combined[col].apply(lambda v: f"{v:,.0f}" if pd.notna(v) else v)
    if "yield_pct" in combined.columns:
        combined["yield_pct"] = combined["yield_pct"].apply(lambda v: f"{v:.2f}%" if pd.notna(v) else v)

    combined.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"تم الحفظ: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
