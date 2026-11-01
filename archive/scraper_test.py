"""
سكربت سحب بيانات العقارات من aqar.fm
- يسحب صفحات القوائم (listing pages) لاستخراج روابط الإعلانات
- يفتح كل إعلان ويحاول استخراج البيانات الكاملة (بما فيها الصور والإحداثيات)
  عبر قراءة الـ JSON المضمّن بالصفحة (__NEXT_DATA__) إن وجد، وإلا يرجع لـ HTML parsing
- يحفظ النتائج بملف CSV، ويتجنب تكرار الإعلانات (dedupe حسب رقم الإعلان الفريد)
"""

import requests
from bs4 import BeautifulSoup
import json
import csv
import time
import re
import os
from urllib.parse import urljoin, unquote

BASE_URL = "https://sa.aqar.fm"

# ==== عدّل هذي القائمة حسب احتياجك (مدينة/نوع عقار/عدد صفحات) ====
# نسحب منطقة وحدة كاملة بكل مرة عشان نتحكم بالوقت والموارد.
# بعد ما تخلص شمال الرياض، غيّر الرابط التالي لمنطقة ثانية (شرق-الرياض، غرب-الرياض...)
LIST_PAGES = [
    "https://sa.aqar.fm/شقق-للبيع/الرياض/شمال-الرياض",
]
MAX_PAGES_PER_CATEGORY = 1   # اختبار سريع بصفحة وحدة بس   # سقف أعلى من الحاجة الفعلية؛ السكربت يتوقف تلقائيًا عند آخر صفحة فعلية

# مسارات محظورة صراحة بـ robots.txt -- لازم نتجنبها دائمًا
FORBIDDEN_PATH_PREFIXES = [
    "/contact-us", "/اتصل-بنا", "/معلومات-المعلن", "/contact_user",
    "/send_iphone", "/send_android", "/download_app",
    "/search/", "/regions/", "/view/", "/map/", "/map-ad/",
    "/district/", "/direction/", "/city/",
    "/add-listing/", "/add-rega-listing/", "/editlisting/",
    "/user/bookings", "/financing/application", "/login",
    "/graphql", "/auth-graphql",
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept-Language": "ar,en;q=0.8",
}

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUTPUT_CSV = os.path.join(DATA_DIR, "listings.csv")

CSV_FIELDS = [
    "listing_id", "url", "title", "price", "area_sqm",
    "rooms", "bathrooms", "livings", "age_years", "district", "city", "direction",
    "description", "latitude", "longitude", "images", "images_count",
    "advertiser_name", "advertiser_company", "advertiser_type",
    "created_at", "published_at", "last_update", "views", "date_scraped",
]

IMAGE_BASE_URL = "https://images.aqar.fm/webp/750x0/props/"


def is_forbidden(path: str) -> bool:
    return any(path.startswith(p) for p in FORBIDDEN_PATH_PREFIXES)


def parse_city_direction_from_url(url):
    """يستخرج اسم المدينة والاتجاه من مسار الرابط نفسه (أوثق من الـ JSON).
    مثال: /شقق-للبيع/الرياض/شمال-الرياض/حي-الياسمين/... -> (الرياض, شمال الرياض)"""
    path = unquote(url.replace(BASE_URL, "")).strip("/")
    parts = path.split("/")
    city = parts[1].replace("-", " ") if len(parts) > 1 else None
    direction = parts[2].replace("-", " ") if len(parts) > 2 else None
    return city, direction


def extract_listing_id(url: str) -> str:
    """الرقم التعريفي دائمًا آخر أرقام بنهاية الرابط"""
    match = re.search(r"-(\d+)/?$", url)
    return match.group(1) if match else url


def get_soup(url: str):
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")


NEXT_F_PATTERN = re.compile(r'self\.__next_f\.push\(\[1,"((?:[^"\\]|\\.)*)"\]\)', re.DOTALL)


def _unescape_js_string(raw):
    """يفك ترميز نص JS المهرّب (يستخدم نفس قواعد تهريب JSON)."""
    return json.loads('"' + raw + '"')


def extract_rsc_text(html):
    """يجمع كل أجزاء self.__next_f.push(...) بالصفحة في نص واحد مفكوك الترميز."""
    parts = []
    for m in NEXT_F_PATTERN.finditer(html):
        try:
            parts.append(_unescape_js_string(m.group(1)))
        except (json.JSONDecodeError, TypeError):
            continue
    return "".join(parts)


def extract_balanced_json(text, anchor):
    """يستخرج كائن JSON متوازن الأقواس يبدأ بعد أول '{' تالي لـ anchor."""
    idx = text.find(anchor)
    if idx == -1:
        return None
    start = text.find("{", idx)
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    i = start
    while i < len(text):
        c = text[i]
        if in_string:
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                in_string = False
        else:
            if c == '"':
                in_string = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
        i += 1
    return None


def resolve_text_reference(rsc_text, ref):
    """يحل مرجع نصي بصيغة '$52' يشير لجزء نص منفصل بالصيغة 'ID:Thex_len,النص'.
    الطول مكتوب بالنظام السداسي عشري ويمثل عدد البايتات (UTF-8) لا عدد الأحرف."""
    m = re.match(r"^\$(\d+)$", ref or "")
    if not m:
        return ref
    chunk_id = m.group(1)
    pattern = re.compile(
        (r"(?:^|\n)" + re.escape(chunk_id) + r":T([0-9a-fA-F]+),").encode("utf-8"),
        re.MULTILINE,
    )
    full_bytes = rsc_text.encode("utf-8")
    match = pattern.search(full_bytes)
    if not match:
        return None
    length = int(match.group(1), 16)
    start = match.end()
    text_bytes = full_bytes[start:start + length]
    try:
        return text_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return None


def extract_listing_json(html):
    """يستخرج كائن 'listing' الصحيح (بيانات العقار) من بيانات RSC المضمّنة بالصفحة،
    وترجع أيضًا نص RSC الكامل (لازم لحل أي مراجع نصية طويلة مثل '$52').
    الصفحة قد تحتوي أكثر من كائن اسمه 'listing' (مثلاً قاموس ترجمة الواجهة)،
    فنفحص كل المطابقات ونختار اللي فيه حقول بيانات العقار الفعلية.
    ترجع (listing_dict, rsc_text) أو (None, rsc_text)."""
    rsc_text = extract_rsc_text(html)
    if not rsc_text:
        return None, ""

    search_from = 0
    while True:
        idx = rsc_text.find('"listing":{', search_from)
        if idx == -1:
            return None, rsc_text
        start = rsc_text.find("{", idx)
        candidate = _extract_balanced_from(rsc_text, start)
        if candidate:
            try:
                parsed = json.loads(candidate)
                # كائن بيانات العقار الحقيقي فيه هذي الحقول، عكس قاموس الترجمة
                if isinstance(parsed, dict) and (
                    "price" in parsed or "imgs" in parsed or "rega_total_price" in parsed
                ):
                    return parsed, rsc_text
            except json.JSONDecodeError:
                pass
        search_from = idx + 1


def _extract_balanced_from(text, start):
    """نفس منطق extract_balanced_json لكن يبدأ من موضع '{' معروف مباشرة."""
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    i = start
    while i < len(text):
        c = text[i]
        if in_string:
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                in_string = False
        else:
            if c == '"':
                in_string = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
        i += 1
    return None


def collect_listing_links_from_list_page(url: str):
    """يسحب صفحة قائمة ويرجع روابط الإعلانات + بيانات أساسية سريعة"""
    soup = get_soup(url)
    links = set()
    for a in soup.select("a[href]"):
        href = a["href"]
        full = urljoin(BASE_URL, href)
        path = full.replace(BASE_URL, "")
        if is_forbidden(path):
            continue
        # روابط الإعلانات تنتهي برقم تعريفي طويل
        if re.search(r"-\d{5,}/?$", full):
            links.add(full)
    return links


def _fmt_timestamp(ts):
    """يحول unix timestamp إلى تاريخ مقروء YYYY-MM-DD، أو يرجع فاضي لو غير موجود."""
    if not ts:
        return None
    try:
        return time.strftime("%Y-%m-%d", time.localtime(int(ts)))
    except (ValueError, TypeError):
        return None


def scrape_listing_detail(url):
    """يفتح صفحة إعلان مفرد ويستخرج كل بياناته من JSON المضمّن بالصفحة (RSC)."""
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    html = resp.text

    data = {
        "listing_id": extract_listing_id(url),
        "url": url,
        "title": None, "price": None, "area_sqm": None,
        "rooms": None, "bathrooms": None, "livings": None, "age_years": None,
        "district": None, "city": None, "direction": None,
        "description": None, "latitude": None, "longitude": None,
        "images": None, "images_count": None,
        "advertiser_name": None, "advertiser_company": None, "advertiser_type": None,
        "created_at": None, "published_at": None, "last_update": None,
        "views": None, "date_scraped": time.strftime("%Y-%m-%d"),
    }

    listing, rsc_text = extract_listing_json(html)

    if listing:
        data["title"] = listing.get("title")
        data["price"] = listing.get("price") or listing.get("rega_total_price")
        data["area_sqm"] = listing.get("area")
        data["rooms"] = listing.get("beds")
        data["bathrooms"] = listing.get("wc")
        data["livings"] = listing.get("livings")
        data["age_years"] = listing.get("age")
        data["district"] = listing.get("district")
        content = listing.get("content")
        if isinstance(content, str) and content.startswith("$"):
            content = resolve_text_reference(rsc_text, content)
        data["description"] = content

        loc = listing.get("location")
        if isinstance(loc, dict):
            data["latitude"] = loc.get("lat")
            data["longitude"] = loc.get("lng")

        imgs = listing.get("imgs") or []
        if isinstance(imgs, list) and imgs:
            full_urls = [IMAGE_BASE_URL + im for im in imgs if isinstance(im, str) and im]
            if full_urls:
                data["images"] = " | ".join(full_urls)
                data["images_count"] = len(full_urls)

        user = listing.get("user")
        if isinstance(user, dict):
            data["advertiser_name"] = user.get("name")
            data["advertiser_company"] = user.get("company_name")
            data["advertiser_type"] = user.get("type")

        data["created_at"] = _fmt_timestamp(listing.get("create_time"))
        data["published_at"] = _fmt_timestamp(listing.get("published_at"))
        data["last_update"] = _fmt_timestamp(listing.get("last_update"))
        data["views"] = listing.get("views")

    # --- خطة احتياطية: لو فشل استخراج الـ JSON بالكامل، نرجع لـ meta tags ---
    # city/direction تُستخرج دائمًا من الرابط، بغض النظر عن نجاح تحليل JSON
    data["city"], data["direction"] = parse_city_direction_from_url(url)

    if not listing:
        soup = BeautifulSoup(html, "html.parser")
        og_title = soup.find("meta", property="og:title")
        if og_title:
            data["title"] = og_title.get("content")
        og_desc = soup.find("meta", property="og:description")
        if og_desc:
            data["description"] = og_desc.get("content")
        og_image = soup.find("meta", property="og:image")
        if og_image:
            data["images"] = og_image.get("content")
            data["images_count"] = 1

    return data


def load_existing_ids():
    if not os.path.exists(OUTPUT_CSV):
        return set()
    with open(OUTPUT_CSV, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return {row["listing_id"] for row in reader}


def append_rows(rows):
    """يُبقى للتوافق، لكن الحفظ الفعلي الآن تدريجي عبر append_row"""
    os.makedirs(DATA_DIR, exist_ok=True)
    file_exists = os.path.exists(OUTPUT_CSV)
    with open(OUTPUT_CSV, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)


def open_csv_writer():
    """يفتح ملف CSV بوضع الإضافة، ويكتب العنوان لو الملف جديد.
    يرجع (file_handle, writer) — لازم تسكر الملف يدويًا بنهاية الاستخدام."""
    os.makedirs(DATA_DIR, exist_ok=True)
    file_exists = os.path.exists(OUTPUT_CSV)
    f = open(OUTPUT_CSV, "a", newline="", encoding="utf-8-sig")
    writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
    if not file_exists:
        writer.writeheader()
        f.flush()
    return f, writer


def main():
    existing_ids = load_existing_ids()
    print(f"عدد الإعلانات المحفوظة مسبقًا: {len(existing_ids)}")

    all_links = set()
    for base in LIST_PAGES:
        print(f"=== تصنيف: {base} ===")
        for page_num in range(1, MAX_PAGES_PER_CATEGORY + 1):
            page_url = base if page_num == 1 else f"{base}/{page_num}"
            try:
                links = collect_listing_links_from_list_page(page_url)
            except requests.RequestException as e:
                print(f"تخطي {page_url}: {e}")
                continue
            if not links:
                print(f"وصلنا آخر صفحة عند صفحة {page_num - 1}، ننتقل للتصنيف التالي")
                break  # وصلنا آخر صفحة متاحة لهذا التصنيف
            print(f"صفحة {page_num}: لقيت {len(links)} رابط (إجمالي حتى الآن: {len(all_links) + len(links)})")
            all_links.update(links)
            time.sleep(2)  # احترام السيرفر

    new_links = [l for l in all_links if extract_listing_id(l) not in existing_ids]
    print(f"روابط جديدة للسحب: {len(new_links)}")

    # --- حفظ تدريجي: كل إعلان يُكتب بالملف فور سحبه، مو مجمّع بالنهاية ---
    # هذا يحمي التقدم لو انقطع التشغيل لأي سبب (بدل ما نخسر كل شي)
    f, writer = open_csv_writer()
    saved_count = 0
    try:
        for link in new_links:
            try:
                row = scrape_listing_detail(link)
                writer.writerow(row)
                f.flush()  # نضمن الكتابة الفعلية على القرص فورًا
                saved_count += 1
                print(f"تم ({saved_count}/{len(new_links)}):", row["listing_id"], row.get("title"))
            except requests.RequestException as e:
                print(f"فشل سحب {link}: {e}")
            time.sleep(2)  # احترام السيرفر بين الطلبات
    finally:
        f.close()

    if saved_count:
        print(f"تمت إضافة {saved_count} إعلان جديد إلى {OUTPUT_CSV}")
    else:
        print("لا توجد إعلانات جديدة اليوم.")


if __name__ == "__main__":
    main()
