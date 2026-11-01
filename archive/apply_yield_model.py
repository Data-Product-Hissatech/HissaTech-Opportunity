"""
يطبّق نموذج توقع الإيجار (المدرّب بـ train_rent_model.py) على عقارات البيع،
يحسب العائد الاستثماري المتوقع لكل عقار = (الإيجار المتوقع ÷ سعر البيع) × 100
ويطلع قائمة نهائية مرتبة بأفضل الفرص (عائد فوق 6%).
"""

import pandas as pd
import numpy as np
import re
import os
import joblib

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# نستخدم بيانات البيع الطبيعية (بعد استبعاد الشذوذ) كأساس، وإلا نرجع للأشمل
SALE_NORMAL_PATH = os.path.join(DATA_DIR, "listings_sale_normal.csv")
SALE_CROSSCHECKED_PATH = os.path.join(DATA_DIR, "listings_sale_area_crosschecked.csv")

MODEL_PATH = os.path.join(DATA_DIR, "rent_model.joblib")
DISTRICT_ENCODING_PATH = os.path.join(DATA_DIR, "rent_model_district_encoding.joblib")

OUTPUT_ALL_PATH = os.path.join(DATA_DIR, "sale_with_predicted_yield.csv")
OUTPUT_OPPORTUNITIES_PATH = os.path.join(DATA_DIR, "top_investment_opportunities.csv")

FURNISHED_PATTERN = re.compile(r"مفروش|مؤثث")

FEATURE_COLS = [
    "area_sqm", "rooms", "bathrooms", "livings", "age_years",
    "latitude", "longitude", "district_encoded", "is_furnished",
]

YIELD_THRESHOLD = 6.0  # النسبة اللي نعتبرها "فرصة جيدة"
MAX_AGE_YEARS = 10  # نستبعد العقارات الأقدم من هذا من قائمة الفرص النهائية


def main():
    if not os.path.exists(MODEL_PATH):
        print(f"تحذير: ما لقيت {MODEL_PATH} -- شغّل train_rent_model.py أول")
        return

    model = joblib.load(MODEL_PATH)
    district_encoding = joblib.load(DISTRICT_ENCODING_PATH)
    print("تم تحميل النموذج وترميز الأحياء")

    sale_path = SALE_NORMAL_PATH if os.path.exists(SALE_NORMAL_PATH) else SALE_CROSSCHECKED_PATH
    print(f"نقرأ بيانات البيع من: {sale_path}")
    df = pd.read_csv(sale_path, encoding="utf-8-sig")
    print(f"عدد عقارات البيع: {len(df)}")

    required_cols = ["area_sqm", "rooms", "bathrooms", "livings", "age_years",
                      "latitude", "longitude", "district", "price"]
    before = len(df)
    df = df.dropna(subset=required_cols).copy()
    print(f"بعد حذف الصفوف الناقصة بأعمدة أساسية: {len(df)} (حذفنا {before - len(df)})")

    # نفس تجهيز الخصائص المستخدم بالتدريب بالضبط
    df["is_furnished"] = df["description"].fillna("").str.contains(FURNISHED_PATTERN).astype(int)
    global_median_rent = district_encoding.median()
    df["district_encoded"] = df["district"].map(district_encoding).fillna(global_median_rent)

    X = df[FEATURE_COLS]
    df["predicted_annual_rent"] = model.predict(X)

    df["expected_yield_pct"] = (df["predicted_annual_rent"] / df["price"] * 100).round(2)

    # سقف منطقي: عائد فوق 20% مستحيل واقعيًا بسوق عقار حقيقي (يعني خطأ بيانات
    # بالسعر غالبًا، مو فرصة حقيقية) -- نستبعده من "الفرص" ونحطه بملف مراجعة منفصل
    MAX_REALISTIC_YIELD = 20.0
    suspicious = df[df["expected_yield_pct"] > MAX_REALISTIC_YIELD]
    df = df[df["expected_yield_pct"] <= MAX_REALISTIC_YIELD]

    print(f"\n⚠️  استبعدنا {len(suspicious)} عقار بعائد مستحيل (فوق {MAX_REALISTIC_YIELD}%) -- على الأغلب خطأ بسعر البيع")
    if len(suspicious):
        suspicious_path = os.path.join(DATA_DIR, "suspicious_yield_needs_price_check.csv")
        suspicious[["listing_id", "url", "district", "price", "area_sqm", "expected_yield_pct"]].to_csv(
            suspicious_path, index=False, encoding="utf-8-sig"
        )
        print(f"    حفظناهم بملف منفصل للمراجعة: {suspicious_path}")

    print(f"\n--- توزيع العائد المتوقع (بعد استبعاد المستحيل) ---")
    print(df["expected_yield_pct"].describe().to_string())

    df_sorted = df.sort_values("expected_yield_pct", ascending=False)

    opportunities = df_sorted[df_sorted["expected_yield_pct"] >= YIELD_THRESHOLD]

    before_age_filter = len(opportunities)
    opportunities = opportunities[opportunities["age_years"] <= MAX_AGE_YEARS]
    print(f"استبعدنا {before_age_filter - len(opportunities)} عقار عمره أكثر من {MAX_AGE_YEARS} سنين من قائمة الفرص")
    print(f"\n🎯 عقارات بعائد متوقع فوق {YIELD_THRESHOLD}%: {len(opportunities)} من {len(df)}")

    cols_to_show = ["listing_id", "url", "district", "price", "area_sqm", "rooms", "age_years",
                     "predicted_annual_rent", "expected_yield_pct"]

    print(f"\n--- أفضل 15 فرصة (أعلى عائد متوقع) ---")
    print(opportunities[cols_to_show].head(15).to_string(index=False))

    df_sorted[cols_to_show].to_csv(OUTPUT_ALL_PATH, index=False, encoding="utf-8-sig")
    opportunities[cols_to_show].to_csv(OUTPUT_OPPORTUNITIES_PATH, index=False, encoding="utf-8-sig")

    print(f"\nتم الحفظ:")
    print(f"  كل عقارات البيع مع العائد المتوقع: {OUTPUT_ALL_PATH}")
    print(f"  الفرص فوق {YIELD_THRESHOLD}% بس: {OUTPUT_OPPORTUNITIES_PATH}")


if __name__ == "__main__":
    main()
