"""
يحدد الأحياء اللي عدد إعلاناتها عندنا قليل بشكل مريب (يحتمل توقف مسحها بدري
بالغلط)، بدون الاعتماد على أي ملف مرجعي خارجي غير موثوق -- يعتمد بس على
توزيع بياناتنا الداخلي (أحياء بعدد أقل بكثير من المتوسط تستاهل إعادة فحص).
"""

import pandas as pd
import os
from urllib.parse import unquote

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
ACTIVE_IDS_PATH = os.path.join(DATA_DIR, "active_ids_today.csv")

BASE_URL = "https://sa.aqar.fm"

# أي حي عدد إعلاناته أقل من هذا الرقم يُعتبر مشبوه ويستاهل إعادة فحص يدوي
SUSPICIOUS_THRESHOLD = 15


def parse_district_direction_from_url(url):
    path = unquote(str(url).replace(BASE_URL, "")).strip("/")
    parts = path.split("/")
    direction = parts[2] if len(parts) > 2 else None
    district = parts[3] if len(parts) > 3 and parts[3].startswith("حي") else None
    return district, direction


def main():
    if not os.path.exists(ACTIVE_IDS_PATH):
        print(f"تحذير: ما لقيت {ACTIVE_IDS_PATH}")
        return

    active = pd.read_csv(ACTIVE_IDS_PATH, encoding="utf-8-sig")
    parsed = active["url"].apply(parse_district_direction_from_url)
    active["district"] = parsed.apply(lambda x: x[0])
    active["direction"] = parsed.apply(lambda x: x[1])

    with_district = active.dropna(subset=["district"])
    counts = with_district.groupby(["direction", "district"]).size().reset_index(name="count")
    counts = counts.sort_values("count")

    suspicious = counts[counts["count"] < SUSPICIOUS_THRESHOLD]
    print(f"إجمالي الأحياء المكتشفة: {len(counts)}")
    print(f"أحياء مشبوهة (أقل من {SUSPICIOUS_THRESHOLD} إعلان): {len(suspicious)}")
    print(f"مجموع إعلاناتها: {suspicious['count'].sum()}")

    print("\n--- الأحياء المشبوهة (رتّبناها من الأقل) ---")
    print(suspicious.to_string(index=False))

    no_district = active["district"].isna().sum()
    print(f"\n⚠️  {no_district} إعلان بدون حي واضح بالرابط (يحتمل مساحة عامة/مشاريع)")

    out_path = os.path.join(DATA_DIR, "suspicious_districts.csv")
    suspicious.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\nتم الحفظ: {out_path}")
    print("\nراجع هالقائمة يدويًا -- افتح رابط كل حي بالمتصفح وقارن العدد المعروض بالموقع مع عندنا")


if __name__ == "__main__":
    main()
