"""
يدمج ملفات الإيجار الخمسة (شمال/شرق/غرب/جنوب/وسط الرياض) بملف واحد نظيف
يزيل التكرار لو أي إعلان تكرر بين الملفات (احتياط، نادر الحدوث)
"""

import pandas as pd
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

REGION_FILES = [
    "listings_rent_north.csv",
    "listings_rent_east.csv",
    "listings_rent_west.csv",
    "listings_rent_south.csv",
    "listings_rent_center.csv",
]


def main():
    dfs = []
    for filename in REGION_FILES:
        path = os.path.join(DATA_DIR, filename)
        if not os.path.exists(path):
            print(f"تحذير: ما لقيت {filename}، نتخطاه")
            continue
        df = pd.read_csv(path, encoding="utf-8-sig")
        print(f"{filename}: {len(df)} صف")
        dfs.append(df)

    if not dfs:
        print("ما فيه أي ملف متاح للدمج!")
        return

    merged = pd.concat(dfs, ignore_index=True)
    before_dedup = len(merged)
    merged = merged.drop_duplicates(subset="listing_id", keep="last")
    after_dedup = len(merged)

    if before_dedup != after_dedup:
        print(f"حذفنا {before_dedup - after_dedup} صف مكرر بين الملفات")

    out_path = os.path.join(DATA_DIR, "listings_rent.csv")
    merged.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\nتم الدمج: {len(merged)} صف إجمالي -> {out_path}")


if __name__ == "__main__":
    main()
