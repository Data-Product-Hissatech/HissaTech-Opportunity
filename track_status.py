"""
تتبع حالة الإعلانات النشطة (بيع الرياض) -- سكربت واحد لكل الأنواع.

يقارن IDs الإعلانات النشطة اليوم بالموقع مع آخر حالة محفوظة، ويحدّث جدول حالة
لكل عقار: جديد / نشط / محتمل_محذوف_أو_مباع، مع first_seen و last_seen.

الاستخدام:
    python track_status.py                 # الشقق فقط (sale) -- السلوك القديم
    python track_status.py villa floor     # أنواع محددة
    python track_status.py all             # الشقق + كل الأنواع

المدخلات (من crawl_active_ids.py و crawl_active_ids_for_type.py):
    sale  : data/active_ids_today.csv          -> data/listings_sale_status.csv
    <نوع> : data/active_ids_<نوع>_today.csv    -> data/listings_<نوع>_status.csv
    (listings_sale.csv / listings_<نوع>.csv اختياري: يُستخدم مرة واحدة لتهيئة الجدول
     لو ما فيه جدول حالة سابق)
"""

import os
import sys
import time

import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
TODAY = time.strftime("%Y-%m-%d")

KINDS = {
    # kind: (ملف النشط اليوم, ملف الحالة, الملف الكامل المحفوظ للتهيئة)
    "sale":     ("active_ids_today.csv",          "listings_sale_status.csv",     "listings_sale.csv"),
    "villa":    ("active_ids_villa_today.csv",    "listings_villa_status.csv",    "listings_villa.csv"),
    "floor":    ("active_ids_floor_today.csv",    "listings_floor_status.csv",    "listings_floor.csv"),
    "land":     ("active_ids_land_today.csv",     "listings_land_status.csv",     "listings_land.csv"),
    "building": ("active_ids_building_today.csv", "listings_building_status.csv", "listings_building.csv"),
    "office":   ("active_ids_office_today.csv",   "listings_office_status.csv",   "listings_office.csv"),
}

NEW, ACTIVE, REMOVED = "جديد", "نشط", "محتمل_محذوف_أو_مباع"
COLS = ["listing_id", "url", "status", "first_seen", "last_seen"]

# حماية: لو العدد النشط اليوم أقل من هذي النسبة من النشط سابقًا، نعتبر الزحف فاشل
MIN_RATIO = 0.5
MIN_PREVIOUS_FOR_GUARD = 100


def read_ids(path):
    df = pd.read_csv(path, encoding="utf-8-sig", dtype=str)
    df["listing_id"] = df["listing_id"].astype(str)
    if "url" not in df.columns:
        df["url"] = ""
    return df.drop_duplicates("listing_id")


def load_status(status_path, listings_path):
    if os.path.exists(status_path):
        df = pd.read_csv(status_path, encoding="utf-8-sig", dtype=str)
        df["listing_id"] = df["listing_id"].astype(str)
        for c in COLS:
            if c not in df.columns:
                df[c] = ""
        return df[COLS]
    if os.path.exists(listings_path):
        # أول تشغيل: نعتبر كل المحفوظ "نشط" بتاريخ اليوم، والمختفي منه يتعلّم محذوف
        saved = read_ids(listings_path)
        saved["status"], saved["first_seen"], saved["last_seen"] = ACTIVE, TODAY, TODAY
        print(f"  (تهيئة من {os.path.basename(listings_path)}: {len(saved):,} إعلان، "
              f"first_seen = اليوم لأن التاريخ الحقيقي غير معروف)")
        return saved[COLS]
    return pd.DataFrame(columns=COLS)


def track(kind):
    today_file, status_file, listings_file = KINDS[kind]
    today_path = os.path.join(DATA_DIR, today_file)
    status_path = os.path.join(DATA_DIR, status_file)
    listings_path = os.path.join(DATA_DIR, listings_file)

    print(f"\n=== {kind} ===")
    if not os.path.exists(today_path):
        print(f"  ما لقيت {today_file} -- شغّل الزحف أول، أتخطى {kind}")
        return False

    today_df = read_ids(today_path)
    today_ids = set(today_df["listing_id"])
    print(f"  نشط اليوم بالموقع: {len(today_ids):,}")

    status = load_status(status_path, listings_path)
    prev_active = set(status.loc[status["status"].isin([NEW, ACTIVE]), "listing_id"])

    if len(prev_active) >= MIN_PREVIOUS_FOR_GUARD and len(today_ids) < MIN_RATIO * len(prev_active):
        print(f"  ⚠ العدد اليوم ({len(today_ids):,}) أقل بكثير من النشط سابقًا ({len(prev_active):,}). "
              f"غالبًا الزحف فشل أو انقطع، فما حدّثت الحالة لتجنب تعليم الكل محذوف.")
        return False

    known = set(status["listing_id"])
    new_ids = today_ids - known                      # لأول مرة نشوفه
    returned_ids = (today_ids & known) - prev_active # كان محذوف ورجع
    still_ids = today_ids & prev_active
    removed_ids = prev_active - today_ids

    print(f"  جديد: {len(new_ids):,} | لسا نشط: {len(still_ids):,} | "
          f"رجع بعد اختفاء: {len(returned_ids):,} | محتمل محذوف/مباع: {len(removed_ids):,}")

    status = status.set_index("listing_id")

    # الموجود اليوم (نشط أو رجع): last_seen = اليوم، والحالة نشط
    seen_existing = list((still_ids | returned_ids) & set(status.index))
    status.loc[seen_existing, "status"] = ACTIVE
    status.loc[seen_existing, "last_seen"] = TODAY

    # المختفي: نعلّمه ونحتفظ بتاريخه
    gone = list(removed_ids & set(status.index))
    status.loc[gone, "status"] = REMOVED

    # الجديد
    if new_ids:
        urls = today_df.set_index("listing_id")["url"]
        add = pd.DataFrame({
            "url": [urls.get(i, "") for i in new_ids],
            "status": NEW,
            "first_seen": TODAY,
            "last_seen": TODAY,
        }, index=pd.Index(list(new_ids), name="listing_id"))
        status = pd.concat([status, add])

    # نحدّث رابط من كان رابطه فاضي
    urls = today_df.set_index("listing_id")["url"]
    empty = status["url"].fillna("") == ""
    status.loc[empty, "url"] = status.index[empty].map(lambda i: urls.get(i, ""))

    status.reset_index()[COLS].to_csv(status_path, index=False, encoding="utf-8-sig")
    print(f"  تم الحفظ: {status_file} ({len(status):,} سجل)")
    return True


def main():
    args = [a.lower() for a in sys.argv[1:]] or ["sale"]
    kinds = list(KINDS) if "all" in args else args
    bad = [k for k in kinds if k not in KINDS]
    if bad:
        print(f"نوع غير معروف: {bad}. المتاح: {list(KINDS)} أو all")
        sys.exit(2)
    ok = [track(k) for k in kinds]
    print(f"\nانتهى: {sum(ok)}/{len(kinds)} نجح")


if __name__ == "__main__":
    main()
