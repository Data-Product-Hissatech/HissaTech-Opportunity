# -*- coding: utf-8 -*-

"""
HissaTech Saudi Real Estate Deal Analyzer

Pipeline:
1) Read sale listings CSV.
2) Cheap Regex filter for listings that mention an active tenant/rent.
3) LLM extraction from description using llama-3.3-70b-versatile.
4) Programmatic validation/correction of annual rent.
5) Calculate current gross yield.
6) Classify initial HissaTech result using minimum target yield = 6%.
7) Select best candidates.
8) Use Groq Compound:
   - Visit original listing.
   - Search web for comparable sale listings.
   - Search web for comparable rental listings.
   - Search/use Raghdan when useful.
9) Produce a fixed Arabic HissaTech report template.
10) Save:
    - currently_rented_properties.csv
    - deep_analysis_rented_properties.csv
    - reports/<listing_id>.md
    - research/<listing_id>_tools.json

Required env:
    GROQ_API_KEY

Expected input:
    data/listings_sale_normal.csv

Optional:
    data/sakani_rent_indicators.csv
"""

import os
import re
import json
import time
import math
import urllib.request
import urllib.error
from pathlib import Path

import pandas as pd


# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
REPORTS_DIR = BASE_DIR / "reports"
RESEARCH_DIR = BASE_DIR / "research"

INPUT_PATH = DATA_DIR / "listings_sale_normal.csv"
RENTED_OUTPUT_PATH = DATA_DIR / "currently_rented_properties.csv"
DEEP_OUTPUT_PATH = DATA_DIR / "deep_analysis_rented_properties.csv"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
RESEARCH_DIR.mkdir(parents=True, exist_ok=True)

API_URL = "https://api.groq.com/openai/v1/chat/completions"

EXTRACTION_MODEL = "openai/gpt-oss-120b"  # llama-3.3-70b-versatile قاعد رسميًا (17 يونيو 2026)، هذا البديل الموصى به من Groq
RESEARCH_MODEL = "groq/compound"

HISSATECH_MIN_YIELD = 6.0
EXIT_HORIZON = "3-5 سنوات"

DEEP_ANALYZE_TOP_N = 10
EXTRACTION_DELAY_SECONDS = 3.0  # رُفع من 0.35 -- حد Groq المجاني 8000 توكن/دقيقة، هالتأخير يبقينا تحته
RESEARCH_DELAY_SECONDS = 1.5
MAX_DESCRIPTION_CHARS = 5000
MAX_API_RETRIES = 2


# ============================================================
# Basic helpers
# ============================================================

def is_missing(value):
    try:
        return value is None or pd.isna(value)
    except Exception:
        return value is None


def clean_text(value):
    if is_missing(value):
        return ""
    return str(value).strip()


def safe_float(value):
    if is_missing(value):
        return None
    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isnan(value):
            return None
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    text = (
        text.replace(",", "")
        .replace("٬", "")
        .replace("SAR", "")
        .replace("ريال", "")
        .strip()
    )
    try:
        return float(text)
    except Exception:
        return None


def safe_int(value):
    number = safe_float(value)
    if number is None:
        return None
    return int(round(number))


def money(value):
    number = safe_float(value)
    if number is None:
        return "غير متوفر"
    return f"{number:,.0f}"


def decimal(value, digits=2):
    number = safe_float(value)
    if number is None:
        return None
    return round(number, digits)


def get_row_value(row, *names, default=None):
    for name in names:
        if name in row.index:
            value = row.get(name)
            if not is_missing(value) and str(value).strip() != "":
                return value
    return default


# ============================================================
# Regex candidate filter
# ============================================================

RENTED_HINTS = re.compile(
    r"مؤجرة|مؤجر\b|مؤجّرة|مؤجّر\b|"
    r"يوجد مستأجر|مستأجر حاليًا|مستأجر حاليا|"
    r"عقد إيجار ساري|عقد ايجار ساري|"
    r"عقد إيجار سنوي|عقد ايجار سنوي|"
    r"دخل ثابت|مؤجرة سنوي|مؤجر سنوي",
    re.IGNORECASE,
)


# ============================================================
# Groq API
# ============================================================

def groq_request(
    api_key,
    model,
    messages,
    max_tokens=2000,
    temperature=0.1,
    enable_web_tools=False,
):
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_completion_tokens": max_tokens,
    }

    if enable_web_tools:
        payload["compound_custom"] = {
            "tools": {
                "enabled_tools": [
                    "web_search",
                    "visit_website",
                    "code_interpreter",
                ]
            }
        }

    encoded_payload = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        # مهم: بدون هذا الرأس، urllib.request يرسل "Python-urllib/3.x"
        # كـUser-Agent افتراضي -- Cloudflare (اللي يحمي Groq) يحظره كبصمة
        # بوت مشبوهة (خطأ 1010). نرسل بصمة متصفح حقيقية بدلها.
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    }

    if enable_web_tools:
        headers["Groq-Model-Version"] = "latest"

    last_error = None
    rate_limit_wait = None  # لو صار 429، نحفظ الوقت المقترح من رسالة الخطأ نفسها

    for attempt in range(MAX_API_RETRIES + 1):
        try:
            request = urllib.request.Request(
                API_URL,
                data=encoded_payload,
                method="POST",
                headers=headers,
            )
            with urllib.request.urlopen(request, timeout=180) as response:
                raw = response.read().decode("utf-8")
                data = json.loads(raw)

            message = data["choices"][0]["message"]

            return {
                "content": message.get("content", ""),
                "executed_tools": message.get("executed_tools", []),
                "raw_response": data,
            }

        except urllib.error.HTTPError as exc:
            try:
                body = exc.read().decode("utf-8")
            except Exception:
                body = ""
            last_error = RuntimeError(f"Groq HTTP {exc.code}: {body[:1000]}")

            # خطأ 429 (تجاوز حد الاستخدام) -- نستخرج وقت الانتظار المقترح
            # من رسالة الخطأ نفسها ("Please try again in X.XXs")، وننتظره
            # بدقة بدل الانتظار القصير الثابت العادي
            if exc.code == 429:
                wait_match = re.search(r"try again in ([\d.]+)s", body)
                if wait_match:
                    rate_limit_wait = float(wait_match.group(1)) + 1.0  # هامش أمان
                else:
                    rate_limit_wait = 15.0  # احتياطي لو ما لقينا الرقم بالرسالة

        except Exception as exc:
            last_error = exc

        if attempt < MAX_API_RETRIES:
            wait_seconds = rate_limit_wait if rate_limit_wait else 2 ** attempt
            print(f"  API retry {attempt + 1}/{MAX_API_RETRIES} after {wait_seconds:.1f}s...")
            time.sleep(wait_seconds)
            rate_limit_wait = None  # نصفّرها لو المحاولة الجاية نجحت أو صار خطأ ثاني

    raise last_error


def parse_json_response(text):
    cleaned = clean_text(text)
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        return json.loads(cleaned)
    except Exception:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        return json.loads(cleaned[start:end + 1])

    raise ValueError("Could not find valid JSON in LLM response.")


# ============================================================
# Stage 2 - Description extraction prompt
# ============================================================

EXTRACTION_SYSTEM_PROMPT = """
أنت محلل بيانات عقارية سعودي دقيق جدًا.

مهمتك هنا ليست تقييم الصفقة أو البحث في الإنترنت.

مهمتك فقط:
- قراءة وصف إعلان البيع.
- استخراج البيانات المذكورة صراحة.
- كشف التعارضات الواضحة بين بيانات الإعلان والوصف.
- تحديد هل العقار مؤجر حاليًا.
- استخراج الإيجار السنوي الحقيقي.
- استخراج أهم المواصفات والمخاطر النصية.

ممنوع اختلاق معلومة غير موجودة.

إذا لم تكن معلومة مذكورة بوضوح، استخدم null أو false حسب الحقل.

أرجع JSON صالح فقط بدون Markdown.
""".strip()


EXTRACTION_PROMPT_TEMPLATE = """
بيانات الإعلان المسجلة:

- رقم الإعلان: {listing_id}
- نوع العقار المسجل: {property_type}
- الحي: {district}
- المساحة: {area} م²
- عدد الغرف المسجل: {rooms}
- عدد الحمامات: {bathrooms}
- عمر العقار: {age} سنة
- سعر البيع: {price} ريال

نص وصف الإعلان:

\"\"\"
{description}
\"\"\"

قواعد حرجة لاستخراج الإيجار:

1. سعر البيع هو {price} ريال.
   ممنوع اعتبار سعر البيع إيجارًا.

2. "42,000 ريال سنويًا" يعني:
   annual_rent = 42000

3. "80,000 ريال على دفعتين" يعني:
   annual_rent = 80000
   وليس 160000.

4. إذا ورد:
   "75,000 دفعة واحدة / 80,000 دفعتين"
   فهذه خيارات دفع بديلة لنفس السنة.
   لا تجمع الرقمين.
   استخدم قيمة الدفعة الواحدة كمرجع أساسي إن كانت واضحة.

5. إذا الإعلان يقول فقط "مؤجر" بدون رقم إيجار:
   is_currently_rented = true
   annual_rent = null

6. إذا كان الرقم غامضًا:
   annual_rent = null

7. لا تفترض مدة العقد المتبقية إذا لم تذكر.

8. "العمر سنتان" يعني age_years = 2.

9. فرّق بين:
   - عدد غرف النوم
   - المجلس
   - الصالة
   - المقلط
   ولا تغيّر rooms المسجلة إلا إذا كان التعارض واضحًا.

أرجع هذا الهيكل فقط:

{{
  "is_multi_unit": false,

  "data_conflicts": {{
    "district": null,
    "actual_property_type": "شقة",
    "area_sqm": null,
    "rooms": null,
    "bathrooms": null,
    "age_years": null
  }},

  "is_currently_rented": true,
  "annual_rent": 42000,

  "lease_details": "",
  "rent_confidence": "عالية",

  "bedrooms": null,
  "floor": null,

  "key_features": [],
  "risk_flags": [],

  "has_kitchen": false,
  "is_furnished": false,
  "has_private_yard": false,
  "has_private_entrance": false,
  "has_parking": false,
  "has_elevator": false,
  "has_ac": false,

  "is_monthly_rental": false,
  "is_shared_deed": false,
  "is_long_term_lease": false,

  "deed_area_sqm": null,
  "license_number": null,
  "plan_plot_reference": null
}}
""".strip()


# ============================================================
# Programmatic annual-rent correction
# ============================================================

def parse_amount_token(raw_number, context):
    amount = safe_float(raw_number)
    if amount is None:
        return None
    if amount < 1000 and re.search(r"ألف|الف", context):
        amount *= 1000
    return amount


def fix_rent_doubling(annual_rent, description_text, price):
    annual_rent = safe_float(annual_rent)
    price = safe_float(price)
    description_text = clean_text(description_text)

    alt_payment_pattern = re.search(
        r"([\d,٬.]+)\s*(?:ألف|الف)?\s*(?:ريال)?"
        r"\s*(?:دفعة|دفعه)\s*(?:واحدة|واحده)?"
        r"\s*(?:و|/|\\|-)\s*"
        r"([\d,٬.]+)\s*(?:ألف|الف)?\s*(?:ريال)?"
        r"\s*دفعتين",
        description_text,
    )

    if alt_payment_pattern:
        context = description_text[
            max(0, alt_payment_pattern.start() - 20):
            alt_payment_pattern.end() + 20
        ]
        amount1 = parse_amount_token(alt_payment_pattern.group(1), context)
        amount2 = parse_amount_token(alt_payment_pattern.group(2), context)

        if amount1 and amount2:
            correct_amount = min(amount1, amount2)
            return correct_amount

    installment_pattern = re.search(
        r"([\d,٬.]+)\s*(?:ألف|الف)?\s*(?:ريال)?"
        r"\s*(?:سنوي(?:اً|ا)?\s*)?"
        r"(?:على\s*)?"
        r"(?:دفعتين|كل\s*6\s*(?:أشهر|شهور)|نصف\s*سنوي)",
        description_text,
    )

    if installment_pattern:
        context = description_text[
            max(0, installment_pattern.start() - 20):
            installment_pattern.end() + 20
        ]
        stated_amount = parse_amount_token(installment_pattern.group(1), context)

        if stated_amount:
            if annual_rent is None:
                return stated_amount
            if abs(annual_rent - stated_amount * 2) <= max(1000, stated_amount * 0.05):
                print(f"  Rent correction: {annual_rent:,.0f} -> {stated_amount:,.0f}")
                return stated_amount

    if annual_rent and price:
        suspicious_ratios = [1.0, 0.75, 0.5, 0.25, 0.90, 0.95]
        for ratio in suspicious_ratios:
            target = price * ratio
            if abs(annual_rent - target) / price < 0.03:
                print(f"  Rejected suspicious rent {annual_rent:,.0f} vs price {price:,.0f}")
                return None

        yield_pct = annual_rent / price * 100
        if yield_pct > 20:
            print(f"  Rejected implausible gross yield: {yield_pct:.2f}%")
            return None

    return annual_rent


# ============================================================
# Initial HissaTech classification
# ============================================================

def compute_initial_verdict(yield_pct, has_reliable_number):
    yield_pct = safe_float(yield_pct)

    if not has_reliable_number or yield_pct is None:
        return ("Review", "بيانات ناقصة - لا يوجد رقم إيجار موثوق كافٍ للتقييم.")

    if yield_pct >= HISSATECH_MIN_YIELD:
        return (
            "Proceed",
            f"العائد الإجمالي الحالي {yield_pct:.2f}% يحقق أو يتجاوز مستهدف حصتك (≥{HISSATECH_MIN_YIELD:.1f}%)."
        )

    if yield_pct >= HISSATECH_MIN_YIELD - 1.0:
        return (
            "Review",
            f"العائد الإجمالي الحالي {yield_pct:.2f}% قريب من مستهدف حصتك ({HISSATECH_MIN_YIELD:.1f}%)."
        )

    return (
        "Reject",
        f"العائد الإجمالي الحالي {yield_pct:.2f}% أقل بوضوح من مستهدف حصتك ({HISSATECH_MIN_YIELD:.1f}%)."
    )


# ============================================================
# Stage 2 execution
# ============================================================

def extract_listing(row, api_key):
    listing_id = clean_text(get_row_value(row, "listing_id", "id", default=""))
    url = clean_text(get_row_value(row, "url", "listing_url", default=""))
    description = clean_text(get_row_value(row, "description", default=""))
    district = clean_text(get_row_value(row, "district", default=""))
    property_type = clean_text(get_row_value(row, "property_type", "type", default="شقة"))
    area = safe_float(get_row_value(row, "area_sqm", "area", default=None))
    rooms = safe_float(get_row_value(row, "rooms", default=None))
    bathrooms = safe_float(get_row_value(row, "bathrooms", default=None))
    age = safe_float(get_row_value(row, "age_years", "age", default=None))
    price = safe_float(get_row_value(row, "price", default=None))

    prompt = EXTRACTION_PROMPT_TEMPLATE.format(
        listing_id=listing_id or "غير متوفر",
        property_type=property_type or "شقة",
        district=district or "غير متوفر",
        area=area if area is not None else "غير متوفر",
        rooms=rooms if rooms is not None else "غير متوفر",
        bathrooms=(bathrooms if bathrooms is not None else "غير متوفر"),
        age=age if age is not None else "غير متوفر",
        price=price if price is not None else "غير متوفر",
        description=description[:MAX_DESCRIPTION_CHARS],
    )

    response = groq_request(
        api_key=api_key,
        model=EXTRACTION_MODEL,
        messages=[
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        max_tokens=1800,
        temperature=0.0,
        enable_web_tools=False,
    )

    parsed = parse_json_response(response["content"])
    conflicts = parsed.get("data_conflicts") or {}

    actual_area = safe_float(conflicts.get("area_sqm")) or area
    actual_rooms = safe_float(conflicts.get("rooms")) or rooms
    actual_bathrooms = safe_float(conflicts.get("bathrooms")) or bathrooms
    actual_age = safe_float(conflicts.get("age_years"))
    if actual_age is None:
        actual_age = age

    actual_district = clean_text(conflicts.get("district")) or district
    actual_property_type = clean_text(conflicts.get("actual_property_type")) or property_type or "شقة"

    annual_rent = fix_rent_doubling(parsed.get("annual_rent"), description, price)

    yield_pct = None
    rent_multiple = None
    price_per_sqm = None

    if price and actual_area:
        price_per_sqm = round(price / actual_area)

    if annual_rent and price:
        yield_pct = round(annual_rent / price * 100, 2)
        rent_multiple = round(price / annual_rent, 2)

    has_reliable_rent = (
        annual_rent is not None
        and parsed.get("rent_confidence") != "لا يوجد رقم"
    )

    verdict, verdict_reason = compute_initial_verdict(yield_pct, has_reliable_rent)

    key_features = parsed.get("key_features") or []
    if isinstance(key_features, list):
        key_features_text = " | ".join(clean_text(x) for x in key_features if clean_text(x))
    else:
        key_features_text = clean_text(key_features)

    risk_flags = parsed.get("risk_flags") or []
    if isinstance(risk_flags, list):
        risk_flags_text = " | ".join(clean_text(x) for x in risk_flags if clean_text(x))
    else:
        risk_flags_text = clean_text(risk_flags)

    return {
        "listing_id": listing_id,
        "url": url,
        "title": clean_text(get_row_value(row, "title", default="")),
        "district": actual_district,
        "direction": clean_text(get_row_value(row, "direction", default="")),
        "price": price,
        "area_sqm": actual_area,
        "rooms": actual_rooms,
        "bathrooms": actual_bathrooms,
        "age_years": actual_age,
        "actual_property_type": actual_property_type,
        "is_multi_unit": bool(parsed.get("is_multi_unit", False)),
        "is_currently_rented": bool(parsed.get("is_currently_rented", False)),
        "actual_annual_rent": annual_rent,
        "yield_pct_actual": yield_pct,
        "rent_multiple_years": rent_multiple,
        "price_per_sqm": price_per_sqm,
        "lease_details": clean_text(parsed.get("lease_details")),
        "rent_confidence": (clean_text(parsed.get("rent_confidence")) if annual_rent is not None else "لا يوجد رقم"),
        "bedrooms": safe_float(parsed.get("bedrooms")),
        "floor": clean_text(parsed.get("floor")),
        "key_features": key_features_text,
        "risk_flags": risk_flags_text,
        "has_kitchen": bool(parsed.get("has_kitchen", False)),
        "is_furnished": bool(parsed.get("is_furnished", False)),
        "has_private_yard": bool(parsed.get("has_private_yard", False)),
        "has_private_entrance": bool(parsed.get("has_private_entrance", False)),
        "has_parking": bool(parsed.get("has_parking", False)),
        "has_elevator": bool(parsed.get("has_elevator", False)),
        "has_ac": bool(parsed.get("has_ac", False)),
        "is_monthly_rental": bool(parsed.get("is_monthly_rental", False)),
        "is_shared_deed": bool(parsed.get("is_shared_deed", False)),
        "is_long_term_lease": bool(parsed.get("is_long_term_lease", False)),
        "deed_area_sqm": safe_float(parsed.get("deed_area_sqm")),
        "license_number": clean_text(parsed.get("license_number")),
        "plan_plot_reference": clean_text(parsed.get("plan_plot_reference")),
        "initial_verdict": verdict,
        "initial_verdict_reason": verdict_reason,
        "description": description,
    }


# ============================================================
# Candidate review flags
# ============================================================

def build_review_status(row):
    notes = []
    if row.get("actual_property_type") != "شقة":
        notes.append(f"نوع العقار الحقيقي '{row.get('actual_property_type')}' مختلف عن الشقة")
    if bool(row.get("is_monthly_rental")):
        notes.append("الإيجار شهري/متجدد")
    if bool(row.get("is_shared_deed")):
        notes.append("صك مشترك أو تقسيم يحتاج مراجعة")
    if is_missing(row.get("yield_pct_actual")):
        notes.append("لا يوجد إيجار رقمي موثوق")
    if notes:
        return " | ".join(notes)
    return ""


def sort_priority(row):
    has_number = not is_missing(row.get("yield_pct_actual"))
    is_flagged = (
        bool(row.get("is_monthly_rental"))
        or bool(row.get("is_shared_deed"))
        or row.get("actual_property_type") != "شقة"
    )
    is_long_term = bool(row.get("is_long_term_lease"))

    # 0) رقم موثوق + بدون تحذيرات + عقد طويل المدى (سنة+ متبقية) -- الأفضل مطلقًا
    if has_number and not is_flagged and is_long_term:
        return 0
    # 1) رقم موثوق + بدون تحذيرات (بدون تأكيد مدة العقد الطويلة)
    if has_number and not is_flagged:
        return 1
    # 2) رقم موثوق لكن فيه تحذير (شهري/صك مشترك/نوع مختلف)
    if has_number and is_flagged:
        return 2
    # 3) بدون رقم موثوق
    return 3


# ============================================================
# Fixed HissaTech deep-report prompt
# ============================================================

HISSATECH_RESEARCH_SYSTEM_PROMPT = """
أنت محلل صفقات عقارية متخصص في السوق السعودي
وتعمل وفق نموذج استثمار شركة حصتك.

نبذة مختصرة عن حصتك:
حصتك تركز على العقارات المدرة للدخل عبر نموذج
التملك الجزئي، وتهدف إلى اختيار أصول تجمع بين
الدخل الدوري، هامش أمان سعري، وإمكانية التخارج.

المعايير الاستثمارية:
- مستهدف العائد الإجمالي السنوي: 6% أو أكثر.
- فترة التخارج المستهدفة: 3-5 سنوات.
- الاهتمام بالعائد واستدامة الإيجار معًا.
- الاهتمام بقابلية التخارج والنمو الرأسمالي.
- لا تعتبر العائد وحده سببًا كافيًا لـ Proceed.

قواعد النزاهة:
- ممنوع اختلاق إعلان أو سعر أو رابط أو صفقة.
- لا تقل إنك وجدت Comparable إلا إذا ظهر لك فعليًا في البحث أو صفحة ويب تمت زيارتها.
- إذا لم توجد بيانات كافية، قل ذلك صراحة.
- إعلانات البيع والإيجار هي أسعار طلب وليست صفقات منفذة إلا إذا المصدر يثبت خلاف ذلك.
- لا تعامل متوسط الحي العام كـ Comparable مباشر.
- لا تعتبر إعلانين مكررين لنفس العقار مقارنتين.
- انتبه لتعدد الوسطاء وإعادة نشر نفس الوحدة.
- فرّق بوضوح بين: 1) بيانات الإعلان الأصلي 2) إعلانات مقارنة 3) مؤشرات سوق 4) تقدير تحليلي
- لا تدّع أن الحوش أو الموقف أو الصك مستقل نظاميًا ما لم توجد معلومة واضحة تثبت ذلك.
- عند وجود رقم عقد إيجار في الإعلان، اعتبره معلومة من الإعلان وليس إثباتًا قانونيًا للعقد.
- يجب التوصية بالتحقق من العقد والصك عند الحاجة.

المقارنات:
ابحث عن بيع مشابه أولًا في: نفس الحي، نفس نوع العقار، مساحة تقريبًا +/- 20%، عدد غرف قريب، عمر قريب إن أمكن، نفس الشارع أو المشروع له أولوية عالية

وابحث عن إيجارات مشابهة بنفس المنطق.

حاول إيجاد 3-6 مقارنات بيع و3-6 مقارنات إيجار، لكن الجودة أهم من العدد.

النتيجة النهائية يجب أن تكون بالعربية.
""".strip()


REPORT_TEMPLATE_INSTRUCTIONS = """
اكتب التقرير النهائي بنفس الهيكل التالي دائمًا.

ممنوع:
- تغيير أسماء الأقسام.
- حذف قسم.
- إنشاء قسم خامس أو سادس.
- إضافة JSON.
- إضافة مقدمة قبل العنوان.
- إضافة خاتمة بعد التقييم النهائي.

إذا معلومة غير متوفرة اكتب "غير متوفر".
إذا لم تجد مقارنات كافية، صرّح بذلك داخل القسم بدل اختلاق بيانات.

استخدم Markdown.

القالب الإلزامي:

## تقييم الصفقة العقارية — إعلان LISTING_ID

ابدأ بفقرة قصيرة:
"راجعت الإعلان الحالي ومقارنات بيع وإيجار منشورة في DISTRICT..." مع تعديل الجملة إذا لم تتوفر مقارنات كافية.

ثم جدول:

| المؤشر | التقييم |
| --- | --- |
| سعر الشراء | ... |
| المساحة المعلنة | ... |
| المساحة حسب الصك | ... |
| **سعر المتر** | **...** |
| الإيجار الحالي | ... |
| **العائد الإجمالي** | **...** |
| مضاعف الإيجار | ... |
| حالة العقار | ... |
| التوصية | ... |
| الثقة | ... |

### 1. مقارنة سعر الشراء

- اذكر أمثلة المقارنات الفعلية التي عثرت عليها.
- لكل مقارنة مهمة اذكر السعر والمساحة إن توفرت.
- وضح إن كانت أسعار طلب.
- ابحث عن احتمال وجود إعلان آخر لنفس الوحدة.
- إذا وجدت نسخة لنفس الوحدة بسعر آخر، اذكر ذلك.
- احسب/ناقش سعر متر العقار.
- لا تساوي بين متوسط الحي وبين Comparable مباشر.
- ناقش أثر العمر، الدور، الحوش، التجهيزات، المشروع أو الموقع عند توفرها.

### 2. تحليل الإيجار والعائد

- اذكر معادلة العائد الفعلي بالأرقام.
- قارن الإيجار الحالي بعروض إيجار مشابهة.
- حدد هل الإيجار يبدو منطقيًا أو مرتفعًا/منخفضًا.
- وضح أن العائد إجمالي وليس صافيًا.
- اعمل Stress Test محافظ إذا كانت البيانات تسمح.
- لا تخترع نسبة مصاريف كأنها حقيقة. إذا استخدمت افتراض مصروفات، سمّه افتراضًا صراحة.

### 3. نقاط القوة

اكتب نقاط القوة الحقيقية المستندة إلى الإعلان والبحث، بدون مبالغة.

### 4. المخاطر التي تحتاج تحققًا

ركز على: عقد الإيجار، تاريخ البداية والنهاية، السداد والمتأخرات، الصك والفرز، الحوش/الارتداد، الأجزاء المشتركة، اتحاد الملاك، رسوم الصيانة، الالتزامات غير الظاهرة، تعدد الإعلانات بحسب ما ينطبق فعليًا.

اختم القسم بالتنبيه إلى أن مقارنات الإعلانات هي أسعار طلب وليست تقييمًا رسميًا إذا كان ذلك صحيحًا.

### القرار الاستثماري

ابدأ بأحد هذه القرارات فقط:

🟢 **Proceed — فرصة استثمارية جيدة**

أو:

🟡 **Review — تحتاج مراجعة إضافية**

أو:

🔴 **Reject — غير مناسبة حاليًا**

ثم فسّر القرار اعتمادًا على: العائد، السعر، المقارنات، استدامة الإيجار، جودة الأصل، قابلية التخارج خلال 3-5 سنوات

ثم:

**نطاق تفاوض مستهدف:** X–Y ريال.

إذا كان من غير المهني اقتراح نطاق بسبب نقص البيانات:
**نطاق تفاوض مستهدف:** غير متوفر — المقارنات غير كافية.

إذا اقترحت نطاقًا، احسب أثره على العائد بشكل صحيح.

وفي آخر سطر فقط:

**التقييم النهائي: Proceed | الثقة: عالية نسبيًا.**

أو Review / Reject حسب القرار.
""".strip()


def build_deep_research_prompt(row):
    listing_id = clean_text(row.get("listing_id"))
    url = clean_text(row.get("url"))
    district = clean_text(row.get("district"))

    price = safe_float(row.get("price"))
    area = safe_float(row.get("area_sqm"))
    deed_area = safe_float(row.get("deed_area_sqm"))
    annual_rent = safe_float(row.get("actual_annual_rent"))
    yield_pct = safe_float(row.get("yield_pct_actual"))
    price_per_sqm = safe_float(row.get("price_per_sqm"))
    rent_multiple = safe_float(row.get("rent_multiple_years"))

    description = clean_text(row.get("description"))[:MAX_DESCRIPTION_CHARS]

    return f"""
حلل الصفقة التالية.

أولًا:
إذا كان رابط الإعلان عامًا وقابلًا للوصول، زُر الإعلان الأصلي للتحقق من أكبر قدر ممكن من البيانات الحالية.

رابط الإعلان:
{url if url else "غير متوفر"}

ثم ابحث في الويب عن:
1. شقق بيع مشابهة في {district}.
2. شقق إيجار مشابهة في {district}.
3. نتائج أقرب لنفس الشارع إذا كان الشارع واضحًا.
4. مؤشر رغدان للحي/الرياض إذا كان مفيدًا.
5. أي إعادة نشر واضحة لنفس الوحدة.

لا تستخدم إعلانًا كمقارنة إذا شككت أنه نفس الأصل. وضّح احتمال التكرار بدل إدخاله كمقارنة مستقلة.

بياناتنا المستخرجة من الإعلان:

- رقم الإعلان: {listing_id}
- الحي: {district}
- المنطقة/الاتجاه: {clean_text(row.get("direction"))}
- نوع العقار: {clean_text(row.get("actual_property_type"))}
- سعر الشراء: {money(price)} ريال
- المساحة المعلنة: {money(area)} م²
- المساحة حسب الصك المستخرجة من النص: {money(deed_area)} م²
- سعر المتر المحسوب برمجيًا: {money(price_per_sqm)} ريال/م²
- الغرف المسجلة: {row.get("rooms")}
- غرف النوم المستخرجة: {row.get("bedrooms")}
- الحمامات: {row.get("bathrooms")}
- العمر: {row.get("age_years")} سنة
- الدور: {clean_text(row.get("floor")) or "غير متوفر"}

- الإيجار الحالي المستخرج: {money(annual_rent)} ريال/سنة
- العائد الإجمالي المحسوب برمجيًا: {yield_pct if yield_pct is not None else "غير متوفر"}%
- مضاعف الإيجار المحسوب: {rent_multiple if rent_multiple is not None else "غير متوفر"} سنة

- ثقة استخراج الإيجار: {clean_text(row.get("rent_confidence"))}
- تفاصيل العقد: {clean_text(row.get("lease_details")) or "غير مذكور"}
- أهم المميزات: {clean_text(row.get("key_features")) or "غير متوفر"}
- إشارات مخاطر من النص: {clean_text(row.get("risk_flags")) or "لا توجد إشارة واضحة"}
- صك مشترك: {bool(row.get("is_shared_deed"))}
- إيجار شهري: {bool(row.get("is_monthly_rental"))}
- رقم الرخصة المستخرج: {clean_text(row.get("license_number")) or "غير متوفر"}
- مرجع المخطط/القطعة: {clean_text(row.get("plan_plot_reference")) or "غير متوفر"}

الوصف الأصلي:

\"\"\"
{description}
\"\"\"

أرقام محسوبة مسبقًا من Python: لا تغيّرها إلا إذا أثبتت صفحة الإعلان أن بيانات المدخل نفسها مختلفة:

- Price = {price}
- Area = {area}
- Price per sqm = {price_per_sqm}
- Annual rent = {annual_rent}
- Gross yield = {yield_pct}
- Rent multiple = {rent_multiple}

مستهدف حصتك:
- Gross Yield >= {HISSATECH_MIN_YIELD:.1f}%
- Exit Horizon = {EXIT_HORIZON}

مهم جدًا:
ارتفاع العائد وحده لا يعني Proceed تلقائيًا. إذا كان السعر غير مدعوم بالمقارنات، أو الإيجار غير مستدام، أو التخارج ضعيف، خذ ذلك في القرار.

{REPORT_TEMPLATE_INSTRUCTIONS.replace("LISTING_ID", listing_id).replace("DISTRICT", district)}
""".strip()


# ============================================================
# Deep research
# ============================================================

def deep_research_listing(row, api_key):
    listing_id = clean_text(row.get("listing_id"))
    prompt = build_deep_research_prompt(row)

    response = groq_request(
        api_key=api_key,
        model=RESEARCH_MODEL,
        messages=[
            {"role": "system", "content": HISSATECH_RESEARCH_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        max_tokens=7000,
        temperature=0.15,
        enable_web_tools=True,
    )

    report = clean_text(response["content"])

    report_path = REPORTS_DIR / f"{listing_id}.md"
    with open(report_path, "w", encoding="utf-8") as file:
        file.write(report)
        file.write("\n")

    research_path = RESEARCH_DIR / f"{listing_id}_tools.json"
    with open(research_path, "w", encoding="utf-8") as file:
        json.dump(
            {
                "listing_id": listing_id,
                "url": clean_text(row.get("url")),
                "executed_tools": response["executed_tools"],
            },
            file,
            ensure_ascii=False,
            indent=2,
            default=str,
        )

    final_verdict = "Review"
    confidence = ""

    if "🟢 **Proceed" in report or "التقييم النهائي: Proceed" in report:
        final_verdict = "Proceed"
    elif "🔴 **Reject" in report or "التقييم النهائي: Reject" in report:
        final_verdict = "Reject"
    elif "🟡 **Review" in report or "التقييم النهائي: Review" in report:
        final_verdict = "Review"

    confidence_match = re.search(r"الثقة\s*:\s*\**([^*\n|]+)", report)
    if confidence_match:
        confidence = confidence_match.group(1).strip()

    negotiation_match = re.search(r"\*\*نطاق تفاوض مستهدف:\*\*\s*([^\n]+)", report)
    negotiation_range = negotiation_match.group(1).strip() if negotiation_match else ""

    return {
        "listing_id": listing_id,
        "url": clean_text(row.get("url")),
        "district": clean_text(row.get("district")),
        "price": safe_float(row.get("price")),
        "area_sqm": safe_float(row.get("area_sqm")),
        "price_per_sqm": safe_float(row.get("price_per_sqm")),
        "actual_annual_rent": safe_float(row.get("actual_annual_rent")),
        "current_yield_pct": safe_float(row.get("yield_pct_actual")),
        "rent_multiple_years": safe_float(row.get("rent_multiple_years")),
        "initial_verdict": clean_text(row.get("initial_verdict")),
        "final_verdict": final_verdict,
        "confidence": confidence,
        "negotiation_range": negotiation_range,
        "report_path": str(report_path.relative_to(BASE_DIR)),
        "research_path": str(research_path.relative_to(BASE_DIR)),
    }


# ============================================================
# Pipeline
# ============================================================

def main():
    api_key = os.environ.get("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError("GROQ_API_KEY is missing. Add it to GitHub Actions Secrets.")

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH, encoding="utf-8-sig")

    print("=" * 70)
    print("HissaTech Saudi Real Estate Deal Analyzer")
    print("=" * 70)
    print(f"Input listings: {len(df):,}")

    if "description" not in df.columns:
        raise ValueError("CSV must contain a 'description' column.")

    candidate_mask = df["description"].fillna("").astype(str).str.contains(RENTED_HINTS, na=False)
    candidates = df[candidate_mask].copy()
    print(f"Stage 1 - rent-related candidates: {len(candidates):,}")

    # وضع تجريبي: لو متغير البيئة TEST_LIMIT محدّد، نقتصر على هالعدد بس
    # (يوفّر توكن وقت التجربة، بدل تشغيل كل المرشحين)
    test_limit = os.environ.get("TEST_LIMIT")
    if test_limit:
        candidates = candidates.head(int(test_limit))
        print(f"وضع تجريبي مفعّل -- مقتصر على {len(candidates)} عقار بس")

    extracted_rows = []

    for counter, (_, row) in enumerate(candidates.iterrows(), start=1):
        listing_id = clean_text(get_row_value(row, "listing_id", "id", default="unknown"))
        print(f"[Extraction {counter}/{len(candidates)}] {listing_id}")

        try:
            result = extract_listing(row, api_key)
        except Exception as exc:
            print(f"  FAILED extraction {listing_id}: {exc}")
            continue

        if not result.get("is_currently_rented"):
            print("  Not currently rented -> skipped")
            continue

        extracted_rows.append(result)
        time.sleep(EXTRACTION_DELAY_SECONDS)

    result_df = pd.DataFrame(extracted_rows)
    print(f"Stage 2/3 - confirmed rented listings: {len(result_df):,}")

    if result_df.empty:
        result_df.to_csv(RENTED_OUTPUT_PATH, index=False, encoding="utf-8-sig")
        print("No currently rented properties found.")
        return

    result_df["review_status"] = result_df.apply(build_review_status, axis=1)
    result_df["_sort_priority"] = result_df.apply(sort_priority, axis=1)

    result_df = result_df.sort_values(
        by=["_sort_priority", "yield_pct_actual", "age_years"],
        ascending=[True, False, True],
        na_position="last",
    ).drop(columns=["_sort_priority"])

    result_df.to_csv(RENTED_OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"Saved screening results: {RENTED_OUTPUT_PATH}")

    print("\nInitial verdicts:")
    print(result_df["initial_verdict"].value_counts(dropna=False).to_string())

    deep_candidates = result_df[result_df["yield_pct_actual"].notna()].copy()
    deep_candidates["_clean"] = deep_candidates["review_status"].fillna("").eq("").astype(int)

    deep_candidates = deep_candidates.sort_values(
        by=["_clean", "yield_pct_actual", "age_years"],
        ascending=[False, False, True],
        na_position="last",
    )

    deep_candidates = deep_candidates.head(DEEP_ANALYZE_TOP_N).drop(columns=["_clean"])
    print(f"\nStage 5 - deep research candidates: {len(deep_candidates):,}")

    deep_results = []

    for counter, (_, row) in enumerate(deep_candidates.iterrows(), start=1):
        listing_id = clean_text(row.get("listing_id"))
        print(f"[Research {counter}/{len(deep_candidates)}] {listing_id}")

        try:
            deep_result = deep_research_listing(row, api_key)
            deep_results.append(deep_result)
            print(f"  -> {deep_result['final_verdict']}")
        except Exception as exc:
            print(f"  FAILED research {listing_id}: {exc}")

        time.sleep(RESEARCH_DELAY_SECONDS)

    deep_df = pd.DataFrame(deep_results)

    if not deep_df.empty:
        verdict_rank = {"Proceed": 0, "Review": 1, "Reject": 2}
        deep_df["_rank"] = deep_df["final_verdict"].map(verdict_rank).fillna(9)
        deep_df = deep_df.sort_values(
            by=["_rank", "current_yield_pct"],
            ascending=[True, False],
            na_position="last",
        ).drop(columns=["_rank"])

    deep_df.to_csv(DEEP_OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"Screening CSV : {RENTED_OUTPUT_PATH}")
    print(f"Deep CSV      : {DEEP_OUTPUT_PATH}")
    print(f"Reports       : {REPORTS_DIR}")
    print(f"Research logs : {RESEARCH_DIR}")


if __name__ == "__main__":
    main()
