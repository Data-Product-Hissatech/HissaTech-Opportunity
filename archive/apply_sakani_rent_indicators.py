"""
يستخدم مؤشرات إيجار رسمية من سكني/الهيئة العامة للعقار -- متوسط سعر إيجار
لكل حي **حسب عدد الغرف تحديدًا** (غرفتين/3/4)، مع عدد الصفقات لكل فئة --
ويطبّقها على عقارات البيع لحساب العائد الاستثماري الدقيق.

مصدر البيانات: ملف سكني (Q1-Q2 2026)، القيم بالآلاف (22.5 يعني 22,500 ريال).
"""

import pandas as pd
import os
from generate_investment_reports import (
    FURNISHED_PATTERN, extract_strengths, extract_risks, extract_phone,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
SAKANI_PATH = os.path.join(DATA_DIR, "sakani_rent_indicators.csv")
SALE_PATH = os.path.join(DATA_DIR, "listings_sale_normal.csv")
DEALS_PATH = os.path.join(DATA_DIR, "sale_deals_riyadh_2025_2026.csv")
OUTPUT_PATH = os.path.join(DATA_DIR, "yield_from_sakani_indicators.csv")

# أقل عدد صفقات نثق فيه لفئة غرف معينة (زي شرط الثقة بباقي المشروع)
MIN_TRUSTED_DEALS = 10

# نفس منطق compare_sale_to_deals.py -- مساحة قريبة ±25%، وشقق بس
AREA_TOLERANCE_PCT = 0.25
MIN_COMPARABLE_SALE_DEALS = 3


def find_comparable_sale_deals(deals_district, ad_area):
    area_col = "المساحة (متر مربع)"
    type_col = "النوع"
    low = ad_area * (1 - AREA_TOLERANCE_PCT)
    high = ad_area * (1 + AREA_TOLERANCE_PCT)
    comparable = deals_district[(deals_district[area_col] >= low) & (deals_district[area_col] <= high)]
    if type_col in comparable.columns:
        comparable = comparable[comparable[type_col].astype(str).str.contains("شق", na=False)]
    return comparable


def load_sakani_lookup():
    """يبني قاموس: (اسم الحي بصيغة عندنا) -> {rooms: (السعر بالريال، عدد الصفقات)}"""
    df = pd.read_csv(SAKANI_PATH, encoding="utf-8-sig")

    lookup = {}
    for _, row in df.iterrows():
        district_key = "حي " + str(row["الحي"]).strip()  # نطابق صيغة عمود district عندنا
        lookup[district_key] = {
            2: (row["متوسط السعر غرفتين"] * 1000, row["عدد الصفقات غرفتين"]),
            3: (row["متوسط السعر ثلاث غرف"] * 1000, row["عدد الصفقات ثلاث غرف"]),
            4: (row["متوسط السعر اربع غرف"] * 1000, row["عدد الصفقات اربع غرف "]),
        }
    return lookup


def get_expected_rent(lookup, district, rooms):
    """يرجع (الإيجار المتوقع، عدد الصفقات) لحي وعدد غرف معيّن، أو None لو ما توفر"""
    district_data = lookup.get(district)
    if district_data is None:
        return None, None

    rooms_int = int(rooms) if pd.notna(rooms) else None
    if rooms_int not in (2, 3, 4):
        return None, None  # نغطي بس 2/3/4 غرف (المتوفر بالبيانات الرسمية)

    price, count = district_data[rooms_int]
    if pd.isna(price) or pd.isna(count):
        return None, None
    return price, count


def main():
    if not os.path.exists(SAKANI_PATH):
        print(f"تحذير: ما لقيت {SAKANI_PATH} -- ارفع ملف مؤشرات سكني بهذا الاسم لمجلد data/")
        return
    if not os.path.exists(SALE_PATH):
        print(f"تحذير: ما لقيت {SALE_PATH}")
        return

    lookup = load_sakani_lookup()
    print(f"عدد الأحياء بمؤشرات سكني: {len(lookup)}")

    sale = pd.read_csv(SALE_PATH, encoding="utf-8-sig")
    print(f"عدد عقارات البيع: {len(sale)}")

    deals = None
    if os.path.exists(DEALS_PATH):
        deals = pd.read_csv(DEALS_PATH, encoding="utf-8-sig")
        print(f"عدد صفقات البيع الرسمية (2025-2026): {len(deals)}")
    else:
        print(f"تحذير: ما لقيت {DEALS_PATH} -- بنكمل بدون مقارنة سعر البيع")

    results = []
    for _, row in sale.iterrows():
        expected_rent, deals_count = get_expected_rent(lookup, row.get("district"), row.get("rooms"))
        if expected_rent is None:
            continue

        price = row.get("price")
        area = row.get("area_sqm")
        if pd.isna(price) or price <= 0 or pd.isna(area) or area <= 0:
            continue

        yield_pct = round(expected_rent / price * 100, 2)
        is_trusted = deals_count >= MIN_TRUSTED_DEALS

        description = row.get("description")
        is_furnished = bool(FURNISHED_PATTERN.search(str(description or "")))
        strengths = extract_strengths(description)
        risks = extract_risks(row.get("title"), description)
        phone = extract_phone(description)

        # نبدأ بكل أعمدة الصف الأصلي كاملة (كل تفاصيل العقار كما هي)، وبعدين
        # نضيف/نستبدل أعمدة الإيجار والعائد والتحقق الخاصة بمصدر سكني
        result_row = row.to_dict()
        result_row.update({
            "مؤثثة": "نعم" if is_furnished else "لا",
            "رقم_التواصل": phone,
            "strengths": " | ".join(strengths),
            "risks": " | ".join(risks),
            "expected_annual_rent_sakani": round(expected_rent),
            "sakani_deals_count": int(deals_count),
            "sakani_trusted": is_trusted,
            "expected_yield_pct_sakani": yield_pct,
        })

        # مقارنة سعر البيع بصفقات وزارة العدل الرسمية (لو الملف متوفر)
        if deals is not None:
            district_bare = str(row.get("district", "")).replace("حي ", "").strip()
            deals_district = deals[deals["الحي"] == district_bare]
            comparable = find_comparable_sale_deals(deals_district, area)

            if len(comparable) >= MIN_COMPARABLE_SALE_DEALS:
                ad_price_per_sqm = price / area
                comparable_median = comparable["سعر المتر المربع (ريال)"].median()
                ratio = round(ad_price_per_sqm / comparable_median, 2)

                result_row["ad_price_per_sqm"] = round(ad_price_per_sqm)
                result_row["comparable_sale_deals_count"] = len(comparable)
                result_row["comparable_median_price_per_sqm"] = round(comparable_median)
                result_row["price_ratio"] = ratio
                if ratio < 0.6:
                    result_row["verdict_price"] = "🔴 REVIEW"
                elif ratio > 1.15:
                    result_row["verdict_price"] = "🟡 مبالغ فيه"
                else:
                    result_row["verdict_price"] = "🟢 سعر عادل"
            else:
                result_row["ad_price_per_sqm"] = round(price / area)
                result_row["comparable_sale_deals_count"] = len(comparable)
                result_row["comparable_median_price_per_sqm"] = None
                result_row["price_ratio"] = None
                result_row["verdict_price"] = None

        results.append(result_row)

    result_df = pd.DataFrame(results)

    # ترتيب منطقي: الأهم أول (العائد والتحقق)، ثم تفاصيل العقار، والوصف آخر شي
    ordered_cols = [
        "listing_id", "url", "title", "district", "direction",
        "expected_yield_pct_sakani", "sakani_trusted", "sakani_deals_count",
        "expected_annual_rent_sakani",
        "price", "area_sqm", "rooms", "bathrooms", "livings", "age_years",
        "مؤثثة", "رقم_التواصل",
        "ad_price_per_sqm", "comparable_sale_deals_count", "comparable_median_price_per_sqm",
        "price_ratio", "verdict_price",
        "strengths", "risks",
        "description",
    ]
    ordered_cols = [c for c in ordered_cols if c in result_df.columns]
    ordered_cols += [c for c in result_df.columns if c not in ordered_cols]
    result_df = result_df[ordered_cols].sort_values("expected_yield_pct_sakani", ascending=False)
    print(f"\nعدد العقارات اللي حسبنا لها عائد (حي وعدد غرف متوفرين بمؤشرات سكني): {len(result_df)}")

    trusted = result_df[result_df["sakani_trusted"]]
    print(f"منها بثقة عالية (10+ صفقة لنفس الحي وعدد الغرف): {len(trusted)}")

    print(f"\n--- توزيع العائد (بثقة عالية بس) ---")
    print(trusted["expected_yield_pct_sakani"].describe().to_string())

    print(f"\n--- أفضل 30 فرصة (بثقة عالية) ---")
    cols = ["listing_id", "district", "rooms", "price", "expected_annual_rent_sakani",
            "sakani_deals_count", "expected_yield_pct_sakani"]
    print(trusted[cols].head(30).to_string(index=False))

    result_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"\nتم الحفظ: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
