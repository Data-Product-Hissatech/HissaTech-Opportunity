"""
استخراج العقارات المؤجّرة (Regex بحت، بدون أي API، صفر تكلفة) -- سكربت واحد لكل المدن.

يحل محل:
    rented_extracted_no_api.py (شقق الرياض)      rented_other_types_no_api.py (أنواع الرياض الأخرى)
    rented_from_owner_no_api.py (المالك المباشر)  rented_by_city_no_api.py (باقي المدن)
    rented_candidates_no_api.py (فلترة خام)
مع بقاء rented_extracted_no_api.py كمكتبة للدوال (الـ regex المُختبر).

الاستخدام:
    python rented_no_api.py riyadh            # شقق + فلل + أدوار + أراضي + عمائر + مكاتب
    python rented_no_api.py jeddah mecca      # مدينة أو أكثر
    python rented_no_api.py all               # الرياض + كل المدن + ملف مجمّع

المخرجات (في data/):
    rented_<مدينة>_no_api.csv       لكل مدينة (الرياض: rented_riyadh_no_api.csv)
    rented_all_cities_no_api.csv    عند all فقط: كل المدن في ملف واحد
فلتر المالك المباشر: اختر من_المالك_مباشرة == "نعم" في الملف (ما عاد له ملف منفصل).
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rented_extracted_no_api import (  # noqa: E402
    RENTED_HINTS, extract_annual_rent, sanity_check_rent, extract_key_features,
    normalize_for_duplicate_check, is_owner_direct,
)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

CITIES = {
    "riyadh": "الرياض", "medina": "المدينة المنورة", "jeddah": "جدة", "mecca": "مكة المكرمة",
    "dammam": "الدمام", "khobar": "الخبر", "dhahran": "الظهران", "jubail": "الجبيل",
    "ahsa": "الأحساء", "qatif": "القطيف", "jazan": "جازان", "khamis_mushait": "خميس مشيط",
    "abha": "أبها", "buraidah": "بريدة", "taif": "الطائف",
}

# الرياض: ملفات منفصلة لكل نوع (الشقق من listings_sale)
RIYADH_SOURCES = [
    ("listings_sale", "شقة"),
    ("listings_villa", "فيلا"),
    ("listings_floor", "دور"),
    ("listings_building", "عمارة"),
    ("listings_office", "مكتب"),
    ("listings_land", "أرض"),
]

OUTPUT_COLS = [
    "المدينة", "listing_id", "url", "title", "published_at", "نوع_العقار", "district", "direction",
    "price", "area_sqm", "rooms", "bathrooms", "age_years",
    "actual_annual_rent", "yield_pct", "key_features",
    "من_المالك_مباشرة", "owner_detection_reason", "advertiser_name", "description",
]


def read_first(*names):
    """يقرأ أول ملف موجود (يفضّل النسخة _normal بعد كشف الشذوذ)."""
    for name in names:
        path = os.path.join(DATA_DIR, name)
        if os.path.exists(path):
            return pd.read_csv(path, encoding="utf-8-sig", low_memory=False), name
    return None, None


def find_rented(df, type_label=None):
    """المراحل: فلترة تأجير -> استخراج إيجار وعائد -> خصائص -> مالك مباشر."""
    mask = df["description"].fillna("").astype(str).str.contains(RENTED_HINTS, na=False)
    cand = df[mask].copy()
    if cand.empty:
        return cand

    rents = []
    for _, row in cand.iterrows():
        rent, _ = extract_annual_rent(row.get("description"), row.get("price"))
        rents.append(sanity_check_rent(rent, row.get("price")))
    cand["actual_annual_rent"] = rents
    cand["yield_pct"] = cand.apply(
        lambda r: round(r["actual_annual_rent"] / r["price"] * 100, 2)
        if pd.notna(r["actual_annual_rent"]) and r.get("price") else None, axis=1)
    cand["key_features"] = cand["description"].apply(extract_key_features)

    flags, reasons = [], []
    for _, row in cand.iterrows():
        owner, reason = is_owner_direct(row)
        flags.append("نعم" if owner else "لا")
        reasons.append(reason)
    cand["من_المالك_مباشرة"] = flags
    cand["owner_detection_reason"] = reasons

    if type_label:
        cand["نوع_العقار"] = type_label
    elif "property_type" in cand.columns:
        cand["نوع_العقار"] = cand["property_type"]
    return cand


def finalize(cand, city_label):
    """حذف التكرار + الترتيب + التنسيق (نفس سلوك السكربتات القديمة)."""
    cand = cand.copy()
    cand["_norm"] = cand["description"].apply(normalize_for_duplicate_check)
    before = len(cand)
    cand = cand.drop_duplicates(subset="_norm", keep="first").drop(columns=["_norm"])
    print(f"  حذفنا {before - len(cand)} إعلان مكرر")

    cand["المدينة"] = city_label
    cand["_has_yield"] = cand["yield_pct"].notna()
    cand = cand.sort_values(["_has_yield", "yield_pct", "age_years"],
                            ascending=[False, False, True], na_position="last").drop(columns=["_has_yield"])
    return cand[[c for c in OUTPUT_COLS if c in cand.columns]]


def format_for_excel(df):
    df = df.copy()
    for col in ("price", "actual_annual_rent"):
        if col in df.columns:
            df[col] = df[col].apply(lambda v: f"{v:,.0f}" if pd.notna(v) and not isinstance(v, str) else v)
    if "yield_pct" in df.columns:
        df["yield_pct"] = df["yield_pct"].apply(lambda v: f"{v:.2f}%" if pd.notna(v) and not isinstance(v, str) else v)
    return df


def process_riyadh():
    parts = []
    for base, label in RIYADH_SOURCES:
        names = (["listings_sale_normal.csv", "listings_sale.csv"] if base == "listings_sale"
                 else [f"{base}_normal.csv", f"{base}.csv"])
        df, name = read_first(*names)
        if df is None:
            print(f"  {label}: ما لقينا ملف، تخطّينا")
            continue
        found = find_rented(df, label)
        print(f"  {label}: {len(df):,} إعلان من {name} -> مؤجّرة {len(found):,}")
        if len(found):
            parts.append(found)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def process_city(city_key):
    df, name = read_first(f"listings_sale_{city_key}_all_types_normal.csv",
                          f"listings_sale_{city_key}_all_types.csv")
    if df is None:
        print("  ما لقينا ملف المدينة، تخطّينا")
        return pd.DataFrame()
    print(f"  {len(df):,} إعلان من {name}")
    found = find_rented(df)
    print(f"  مؤجّرة مكتشفة: {len(found):,}")
    return found


def run(city_key):
    label = CITIES[city_key]
    print(f"\n=== {label} ({city_key}) ===")
    cand = process_riyadh() if city_key == "riyadh" else process_city(city_key)
    if cand.empty:
        print("  لا يوجد عقار مؤجّر، ما كتبت ملف")
        return None
    rented = finalize(cand, label)
    n_rent = rented["actual_annual_rent"].notna().sum()
    n_owner = (rented["من_المالك_مباشرة"] == "نعم").sum()
    print(f"  النهائي: {len(rented):,} | برقم إيجار: {n_rent:,} | من المالك مباشرة: {n_owner:,}")
    out = os.path.join(DATA_DIR, f"rented_{city_key}_no_api.csv")
    format_for_excel(rented).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"  تم الحفظ: {out}")
    return rented


def main():
    args = [a.lower() for a in sys.argv[1:]]
    if not args:
        print(__doc__)
        sys.exit(2)
    targets = list(CITIES) if "all" in args else args
    bad = [t for t in targets if t not in CITIES]
    if bad:
        print(f"مدينة غير معروفة: {bad}\nالمتاح: {', '.join(CITIES)} أو all")
        sys.exit(2)

    results = [r for r in (run(c) for c in targets) if r is not None]
    if "all" in args and results:
        allc = pd.concat(results, ignore_index=True)
        out = os.path.join(DATA_DIR, "rented_all_cities_no_api.csv")
        format_for_excel(allc).to_csv(out, index=False, encoding="utf-8-sig")
        print(f"\nالملف المجمّع: {out} ({len(allc):,} صف)")


if __name__ == "__main__":
    main()
