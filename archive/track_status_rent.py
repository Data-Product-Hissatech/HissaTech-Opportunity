"""
يقارن قائمة الإعلانات النشطة اليوم (من crawl_active_ids.py) مع آخر حالة محفوظة،
ويحدّث جدول حالة شامل لكل عقار: جديد / نشط / محتمل محذوف
"""

import pandas as pd
import os
import time

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
TODAY_PATH = os.path.join(DATA_DIR, "active_ids_today_rent.csv")
STATUS_PATH = os.path.join(DATA_DIR, "listings_rent_status.csv")

TODAY = time.strftime("%Y-%m-%d")


def main():
    today_df = pd.read_csv(TODAY_PATH, encoding="utf-8-sig")
    today_ids = set(today_df["listing_id"].astype(str))
    print(f"إعلانات نشطة اليوم بالموقع: {len(today_ids)}")

    if os.path.exists(STATUS_PATH):
        status_df = pd.read_csv(STATUS_PATH, encoding="utf-8-sig")
        status_df["listing_id"] = status_df["listing_id"].astype(str)
    else:
        status_df = pd.DataFrame(columns=[
            "listing_id", "url", "status", "first_seen", "last_seen"
        ])

    # الإعلانات اللي كانت "نشط" أو "جديد" بآخر تحديث (يعني كانت معتبرة موجودة)
    previously_active_ids = set(
        status_df[status_df["status"].isin(["جديد", "نشط"])]["listing_id"]
    )

    new_ids = today_ids - previously_active_ids
    still_active_ids = today_ids & previously_active_ids
    removed_ids = previously_active_ids - today_ids

    print(f"جديد اليوم: {len(new_ids)}")
    print(f"لسا نشط: {len(still_active_ids)}")
    print(f"محتمل محذوف/مباع (اختفى من الموقع): {len(removed_ids)}")

    status_df = status_df.set_index("listing_id")

    # نحدّث الجديد
    today_lookup = today_df.set_index(today_df["listing_id"].astype(str))["url"]
    for lid in new_ids:
        status_df.loc[lid] = {
            "url": today_lookup.get(lid, ""),
            "status": "جديد",
            "first_seen": TODAY,
            "last_seen": TODAY,
        }

    # نحدّث اللي لسا نشط (بس last_seen يتحدث)
    for lid in still_active_ids:
        if lid in status_df.index:
            status_df.loc[lid, "status"] = "نشط"
            status_df.loc[lid, "last_seen"] = TODAY

    # نعلّم المحذوف (بدون ما نحذفه من الجدول -- نحتفظ بالتاريخ)
    for lid in removed_ids:
        if lid in status_df.index:
            status_df.loc[lid, "status"] = "محتمل_محذوف_أو_مباع"

    status_df = status_df.reset_index()
    status_df.to_csv(STATUS_PATH, index=False, encoding="utf-8-sig")
    print(f"\nتم تحديث جدول الحالة: {STATUS_PATH}")
    print(f"إجمالي السجلات بالجدول: {len(status_df)}")


if __name__ == "__main__":
    main()
