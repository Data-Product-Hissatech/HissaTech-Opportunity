"""
وحدة مشتركة يستخدمها كل سكربتات تنظيف الإيجار.
الفكرة: ملف "رئيسي" واحد يتراكم عليه كل التعديلات (بدون حذف أي صف أبدًا)،
وعمود exclusion_reason يعلّم كل صف بسبب استبعاده (لو فيه)، فاضي يعني "مقبول".

قاعدة: أول سبب استبعاد يوصل لصف يفوز -- ما نعيد كتابته لو صف عنده سبب سابق،
عشان نعرف بالضبط أول نقطة فشل لكل عقار.
"""

import os
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MASTER_RENT_PATH = os.path.join(DATA_DIR, "listings_rent_master.csv")
RAW_RENT_PATH = os.path.join(DATA_DIR, "listings_rent.csv")


def load_master(raw_path=RAW_RENT_PATH):
    """يفتح الملف الرئيسي، ويدمج معه أي صفوف جديدة من الخام ما وصلته بعد
    (يعني لو listings_rent.csv كبر بعد آخر تشغيلة، الجديد ينضاف للماستر
    بدل ما يتجاهله الملف الرئيسي القديم)."""
    raw_df = pd.read_csv(raw_path, encoding="utf-8-sig") if os.path.exists(raw_path) else None

    if os.path.exists(MASTER_RENT_PATH):
        master_df = pd.read_csv(MASTER_RENT_PATH, encoding="utf-8-sig")
        if "exclusion_reason" not in master_df.columns:
            master_df["exclusion_reason"] = pd.NA

        if raw_df is not None:
            master_df["listing_id"] = master_df["listing_id"].astype(str)
            raw_df["listing_id"] = raw_df["listing_id"].astype(str)
            known_ids = set(master_df["listing_id"])
            new_rows = raw_df[~raw_df["listing_id"].isin(known_ids)].copy()
            if len(new_rows):
                new_rows["exclusion_reason"] = pd.NA
                print(f"[load_master] لقينا {len(new_rows)} صف جديد بالخام ما كان بالماستر -- ندمجهم")
                master_df = pd.concat([master_df, new_rows], ignore_index=True)
        df = master_df
    else:
        df = raw_df if raw_df is not None else pd.DataFrame()
        df["exclusion_reason"] = pd.NA

    if "exclusion_reason" not in df.columns:
        df["exclusion_reason"] = pd.NA
    return df


def save_master(df):
    df.to_csv(MASTER_RENT_PATH, index=False, encoding="utf-8-sig")
    print(f"تم حفظ الملف الرئيسي: {MASTER_RENT_PATH} ({len(df)} صف)")


def tag_exclusion(df, mask, reason):
    """يعلّم الصفوف المطابقة بسبب الاستبعاد -- بس لو ما عندها سبب سابق (أول سبب يفوز)"""
    untagged = df["exclusion_reason"].isna()
    to_tag = mask & untagged
    df.loc[to_tag, "exclusion_reason"] = reason
    return df, int(to_tag.sum())


def print_reconciliation(df, label="الحالة الحالية"):
    """يطبع ملخص سريع: كم مقبول، وكم مستبعد بكل سبب"""
    print(f"\n--- {label} ---")
    print(f"الإجمالي: {len(df)}")
    accepted = df["exclusion_reason"].isna().sum()
    print(f"مقبول (بدون سبب استبعاد): {accepted}")
    reasons = df["exclusion_reason"].value_counts(dropna=True)
    for reason, count in reasons.items():
        print(f"  مستبعد - {reason}: {count}")
