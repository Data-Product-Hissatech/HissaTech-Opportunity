"""
يدرّب نموذج Random Forest يتوقع الإيجار السنوي المناسب لعقار بناءً على خصائصه.
تحسينات عن النسخة الأولى:
- ترميز الحي بمتوسط سعره (Target Encoding) بدل One-Hot -- يعكس قيمة الحي الفعلية
  بعمود واحد قوي، بدل يتشتت على مئات أعمدة نادرة
- ميزة إضافية: هل الشقة مفروشة (تأثير كبير على الإيجار، مستخرجة من الوصف)
"""

import pandas as pd
import numpy as np
import re
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import os
import joblib

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CLEAN_PATH = os.path.join(DATA_DIR, "listings_rent_clean.csv")
NORMAL_PATH = os.path.join(DATA_DIR, "listings_rent_normal.csv")
INPUT_PATH = NORMAL_PATH if os.path.exists(NORMAL_PATH) else CLEAN_PATH
MODEL_PATH = os.path.join(DATA_DIR, "rent_model_gbr.joblib")
DISTRICT_ENCODING_PATH = os.path.join(DATA_DIR, "rent_model_gbr_district_encoding.joblib")

FURNISHED_PATTERN = re.compile(r"مفروش|مؤثث")

FEATURE_COLS = [
    "area_sqm", "rooms", "bathrooms", "livings", "age_years",
    "latitude", "longitude", "district_encoded", "is_furnished",
]
TARGET_COL = "price"


def add_furnished_flag(df):
    df["is_furnished"] = df["description"].fillna("").str.contains(FURNISHED_PATTERN).astype(int)
    return df


def encode_district(df, district_encoding=None):
    """يستبدل اسم الحي برقم يمثل متوسط سعره (Target Encoding)، لكن بتنعيم إحصائي
    (Smoothing) -- الأحياء اللي عندها عينات قليلة تُرجَّح أقرب للمتوسط العام،
    عشان ما يسيطر إعلان شاذ وحيد على توقع الحي كامل."""
    global_median = df[TARGET_COL].median()

    if district_encoding is None:
        SMOOTHING_K = 20  # كل ما زاد، كل ما احتجنا عينات أكثر عشان نثق بمتوسط الحي الخاص
        grouped = df.groupby("district")[TARGET_COL].agg(["median", "count"])
        grouped["smoothed"] = (
            (grouped["count"] * grouped["median"] + SMOOTHING_K * global_median)
            / (grouped["count"] + SMOOTHING_K)
        )
        district_encoding = grouped["smoothed"]
        print(f"\n--- عينة من تنعيم الأحياء (عدد قليل مقابل كثير) ---")
        sample = grouped.sort_values("count").head(3)
        print(sample.to_string())

    df["district_encoded"] = df["district"].map(district_encoding).fillna(global_median)
    return df, district_encoding


def main():
    print(f"نقرأ من: {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH, encoding="utf-8-sig")
    print(f"عدد الصفوف الكلي: {len(df)}")

    # صفر بعمود الحمامات/الغرف يعني "ما عرفنا العدد" (بيانات مجهولة)، مو "فعلاً صفر"
    # (شقة بدون حمام مستحيل أصلاً) -- نعاملها كمجهول عشان ما تعطي النموذج إشارة مصطنعة
    df["bathrooms"] = df["bathrooms"].replace(0, pd.NA)
    df["rooms"] = df["rooms"].replace(0, pd.NA)

    required_cols = ["area_sqm", "rooms", "bathrooms", "livings", "age_years",
                      "latitude", "longitude", "district", TARGET_COL]
    before = len(df)
    df = df.dropna(subset=required_cols)
    print(f"بعد حذف الصفوف الناقصة بأعمدة أساسية: {len(df)} (حذفنا {before - len(df)})")

    df = add_furnished_flag(df)
    print(f"شقق مفروشة: {df['is_furnished'].sum()} من {len(df)}")

    # نقسّم البيانات أول، ونحسب ترميز الحي من التدريب بس (نتفادى تسرّب بيانات الاختبار)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    train_df, district_encoding = encode_district(train_df)
    test_df, _ = encode_district(test_df, district_encoding=district_encoding)

    X_train = train_df[FEATURE_COLS]
    y_train = train_df[TARGET_COL]
    X_test = test_df[FEATURE_COLS]
    y_test = test_df[TARGET_COL]

    print(f"بيانات التدريب: {len(X_train)} | بيانات الاختبار: {len(X_test)}")

    model = GradientBoostingRegressor(
        n_estimators=300,
        max_depth=4,          # أعمق من كذا يسبب Overfitting بـ Gradient Boosting عادة
        learning_rate=0.05,
        min_samples_leaf=5,
        subsample=0.8,        # يستخدم 80% من البيانات لكل شجرة، يقلل Overfitting
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    print(f"\n--- تقييم النموذج (على بيانات الاختبار) ---")
    print(f"متوسط الخطأ المطلق (MAE): {mae:,.0f} ريال")
    print(f"متوسط نسبة الخطأ (MAPE): {mape:.1f}%")
    print(f"R² (نسبة التباين المفسّر): {r2:.3f}")

    importances = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    print(f"\n--- أهمية الخصائص (مرتبة) ---")
    print(importances.to_string())

    joblib.dump(model, MODEL_PATH)
    joblib.dump(district_encoding, DISTRICT_ENCODING_PATH)
    print(f"\nتم حفظ النموذج: {MODEL_PATH}")
    print(f"تم حفظ ترميز الأحياء: {DISTRICT_ENCODING_PATH}")


if __name__ == "__main__":
    main()
