"""
يقارن IDs الإعلانات النشطة اليوم (من crawl_active_ids_for_type.py) بالمحفوظ
بملف listings_<type>.csv الكامل -- يصنّف كل إعلان: جديد اليوم / لسا نشط /
محتمل محذوف أو مباع (اختفى من نتائج اليوم).
"""

import pandas as pd
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

TYPES = ["villa", "floor", "land", "building", "office"]


def track_type(key):
    listings_path = os.path.join(DATA_DIR, f"listings_{key}.csv")
    active_path = os.path.join(DATA_DIR, f"active_ids_{key}_today.csv")

    if not os.path.exists(listings_path):
        print(f"{key}: ما لقيت {listings_path}، نتخطاه")
        return
    if not os.path.exists(active_path):
        print(f"{key}: ما لقيت {active_path} -- شغّل crawl_active_ids_for_type.py أول")
        return

    listings = pd.read_csv(listings_path, encoding="utf-8-sig")
    active_today = pd.read_csv(active_path, encoding="utf-8-sig")

    active_ids = set(active_today["listing_id"].astype(str))
    saved_ids = set(listings["listing_id"].astype(str))

    def classify(listing_id):
        return "نشط" if str(listing_id) in active_ids else "محتمل محذوف/مباع"

    listings = listings.copy()
    listings["status"] = listings["listing_id"].apply(classify)

    new_today_count = len(active_ids - saved_ids)
    still_active_count = (listings["status"] == "نشط").sum()
    possibly_removed_count = (listings["status"] == "محتمل محذوف/مباع").sum()

    print(f"\n=== {key} ===")
    print(f"إجمالي محفوظ سابقًا: {len(saved_ids)}")
    print(f"نشط بالموقع اليوم: {len(active_ids)}")
    print(f"جديد اليوم (بالنشط، مو بالمحفوظ): {new_today_count}")
    print(f"لسا نشط (من المحفوظ): {still_active_count}")
    print(f"محتمل محذوف/مباع: {possibly_removed_count}")

    output_path = os.path.join(DATA_DIR, f"listings_{key}_status.csv")
    listings.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"تم الحفظ: {output_path}")


def main():
    for key in TYPES:
        track_type(key)


if __name__ == "__main__":
    main()
