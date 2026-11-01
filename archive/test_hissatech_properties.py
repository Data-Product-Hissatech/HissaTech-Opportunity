"""
يختبر عقارات حصتك (يدوية، عيّنة تجريبية) على نفس منهجية النموذج بالضبط:
1. العائد المتوقع من مؤشرات سكني الرسمية (حسب الحي وعدد الغرف)
2. عدالة السعر مقارنة بصفقات وزارة العدل الرسمية المشابهة (نفس الحي والمساحة)

عدّل قائمة PROPERTIES بأسفل بأي عقارات تبي تختبرها.
"""

import pandas as pd
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
SAKANI_PATH = os.path.join(DATA_DIR, "sakani_rent_indicators.csv")
DEALS_PATH = os.path.join(DATA_DIR, "sale_deals_riyadh_2025_2026.csv")

MIN_TRUSTED_DEALS = 10
AREA_TOLERANCE_PCT = 0.25
MIN_COMPARABLE_DEALS = 3
SUSPICIOUS_LOW_RATIO = 0.60
FAIR_HIGH_RATIO = 1.15
YIELD_ACCEPTABLE_THRESHOLD = 5.0  # الحد المطلوب حاليًا (بدل 6/7%)

# ⚠️ عدّل هذي القائمة بأي عقارات تبي تختبرها
PROPERTIES = [
    {"title": "فرصة الافاق",            "district": "حي العارض",  "price": 1180000.00, "area_sqm": 96,  "rooms": 3},
    {"title": "فرصة رواح الملقا",        "district": "حي الملقا",  "price": 1222000.00, "area_sqm": 90,  "rooms": 3},
    {"title": "فرصة حصتك العقارية",      "district": "حي العارض",  "price": 1000000.00, "area_sqm": 120, "rooms": 3},
    {"title": "فرصة النرجس",             "district": "حي النرجس",  "price": 1239000.00, "area_sqm": 120, "rooms": 3},
]
 
def get_expected_rent(sakani, district, rooms):
    district_bare = district.replace("حي ", "").strip()
    match = sakani[sakani["الحي"] == district_bare]
    if len(match) == 0:
        return None, None

    col_map = {2: "متوسط السعر غرفتين", 3: "متوسط السعر ثلاث غرف", 4: "متوسط السعر اربع غرف"}
    count_map = {2: "عدد الصفقات غرفتين", 3: "عدد الصفقات ثلاث غرف", 4: "عدد الصفقات اربع غرف "}

    if rooms not in col_map:
        return None, None

    row = match.iloc[0]
    rent = row[col_map[rooms]] * 1000
    count = row[count_map[rooms]]
    if pd.isna(rent) or pd.isna(count):
        return None, None
    return rent, count


def find_comparable_deals(deals, district, area):
    district_bare = district.replace("حي ", "").strip()
    deals_district = deals[deals["الحي"] == district_bare]
    low = area * (1 - AREA_TOLERANCE_PCT)
    high = area * (1 + AREA_TOLERANCE_PCT)
    comparable = deals_district[
        (deals_district["المساحة (متر مربع)"] >= low) & (deals_district["المساحة (متر مربع)"] <= high)
    ]
    if "النوع" in comparable.columns:
        comparable = comparable[comparable["النوع"].astype(str).str.contains("شق", na=False)]
    return comparable


def classify_price(ratio):
    if ratio < SUSPICIOUS_LOW_RATIO:
        return "🔴 REVIEW"
    elif ratio > FAIR_HIGH_RATIO:
        return "🟡 مبالغ فيه"
    else:
        return "🟢 سعر عادل"


def main():
    sakani = pd.read_csv(SAKANI_PATH, encoding="utf-8-sig")
    deals = pd.read_csv(DEALS_PATH, encoding="utf-8-sig")
    print(f"عدد صفقات وزارة العدل: {len(deals)}\n")

    results = []
    for prop in PROPERTIES:
        rent, deals_count = get_expected_rent(sakani, prop["district"], prop["rooms"])
        yield_pct = round(rent / prop["price"] * 100, 2) if rent else None
        is_trusted = deals_count >= MIN_TRUSTED_DEALS if deals_count else False

        comparable = find_comparable_deals(deals, prop["district"], prop["area_sqm"])
        ad_price_per_sqm = prop["price"] / prop["area_sqm"]

        if len(comparable) >= MIN_COMPARABLE_DEALS:
            comparable_median = comparable["سعر المتر المربع (ريال)"].median()
            ratio = round(ad_price_per_sqm / comparable_median, 2)
            verdict_price = classify_price(ratio)
        else:
            comparable_median = None
            ratio = None
            verdict_price = "غير كافٍ للمقارنة"

        yield_ok = "✅" if yield_pct and yield_pct >= YIELD_ACCEPTABLE_THRESHOLD else "❌"

        results.append({
            "العقار": prop["title"],
            "الحي": prop["district"],
            "السعر": prop["price"],
            "الإيجار المتوقع (سكني)": round(rent) if rent else None,
            "صفقات سكني": int(deals_count) if deals_count else None,
            "موثوق": is_trusted,
            "العائد %": yield_pct,
            f"فوق {YIELD_ACCEPTABLE_THRESHOLD}%؟": yield_ok,
            "سعر متر الإعلان": round(ad_price_per_sqm),
            "صفقات مشابهة (وزارة العدل)": len(comparable),
            "سعر متر مشابه (وسيط)": round(comparable_median) if comparable_median else None,
            "نسبة السعر": ratio,
            "عدالة السعر": verdict_price,
        })

    result_df = pd.DataFrame(results)
    print(result_df.to_string(index=False))

    output_path = os.path.join(DATA_DIR, "hissatech_test_results.csv")
    result_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\nتم الحفظ: {output_path}")


if __name__ == "__main__":
    main()
