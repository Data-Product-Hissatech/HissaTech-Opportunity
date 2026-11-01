"""
يضيف عمودين نهائيين للتقرير (آخر خطوة بالسلسلة، بعد الـ LLM):

1. التوصية_النهائية -- تجمع كل الإشارات المتاحة (العائد + عدالة السعر + تعارضات
   البيانات) بحكم واحد واضح، بدل ما المستخدم يربطها بنفسه.

2. درجة_الثقة -- توضح قوة الحكم بناءً على كم مصدر تحقق متوفر لهذا العقار:
   عالية = عائد موثّق رسميًا + مقارنة صفقات كافية + بدون تعارضات
   متوسطة = مصدر تحقق واحد بس متوفر
   منخفضة = تقدير نموذج فقط، بدون أي تحقق رسمي
"""

import pandas as pd
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
INPUT_PATH = os.path.join(DATA_DIR, "llm_verified_units.csv")
OUTPUT_PATH = os.path.join(DATA_DIR, "final_report_scored.csv")

# حد أدنى من الصفقات المشابهة عشان نعتبر مقارنة السعر "قوية"
STRONG_COMPARISON_MIN_DEALS = 10


def has_conflicts(value):
    """هل الـ LLM اكتشف تعارض بيانات فعلي بهذا الصف"""
    text = str(value or "").strip()
    return text not in ("", "{}", "nan", "None")


def compute_confidence(row):
    """درجة الثقة: كم مصدر تحقق مستقل متوفر لهذا العقار"""
    signals = 0

    # إشارة 1: الإيجار من مصدر رسمي موثّق (حي من الـ11)
    if row.get("ضمن_الـ11_حي_التجريبية") in (True, "True", "TRUE"):
        signals += 1

    # إشارة 2: مقارنة سعر بصفقات رسمية كافية
    deals_count = row.get("comparable_deals_count")
    if pd.notna(deals_count) and deals_count >= STRONG_COMPARISON_MIN_DEALS:
        signals += 1

    # خصم: تعارض بيانات مكتشف يقلل الثقة
    if has_conflicts(row.get("llm_corrections")):
        signals -= 1

    if signals >= 2:
        return "عالية"
    elif signals == 1:
        return "متوسطة"
    else:
        return "منخفضة"


def compute_final_verdict(row):
    """التوصية الموحدة: تجمع العائد + عدالة السعر + التعارضات بحكم واحد"""
    verdict_yield = str(row.get("verdict_yield") or "")
    verdict_price = str(row.get("verdict_price") or "")
    conflicts = has_conflicts(row.get("llm_corrections"))

    # أخطر إشارة: سعر مريب جدًا مقارنة بصفقات حقيقية
    if "REVIEW" in verdict_price:
        return "🔴 لا ينصح بها", "سعر البيع أرخص بشكل غير طبيعي مقارنة بصفقات رسمية مشابهة -- تحقق من الصك والمساحة أولًا"

    # ما عندنا مقارنة سعر أصلاً -- حكم ناقص
    if not verdict_price or verdict_price == "nan":
        return "⚪ غير مكتمل", "ما توفرت صفقات رسمية كافية للتحقق من عدالة السعر -- الحكم مبني على تقدير العائد فقط"

    # سعر مبالغ فيه -- فرصة ضعيفة حتى لو العائد يبدو جيد
    if "مبالغ" in verdict_price:
        return "🟡 تحتاج مراجعة", "سعر البيع أعلى من صفقات رسمية مشابهة -- تفاوض على السعر قبل أي قرار"

    # سعر عادل + تعارض بيانات -- يحتاج تدقيق
    if conflicts:
        return "🟡 تحتاج مراجعة", "السعر عادل، لكن فيه تعارض بين بيانات الإعلان والوصف -- تأكد من التفاصيل الفعلية"

    # سعر عادل + عائد قوي + بدون تعارضات = الحالة المثالية
    if "🟢" in verdict_yield:
        return "🟢 فرصة قوية", "عائد إيجاري قوي، وسعر بيع عادل مقارنة بصفقات رسمية، وبدون أي تعارض بالبيانات"

    return "🟡 تحتاج مراجعة", "سعر عادل وعائد مقبول -- راجع التفاصيل قبل القرار"


def main():
    if not os.path.exists(INPUT_PATH):
        print(f"تحذير: ما لقيت {INPUT_PATH} -- شغّل llm_extract_units.py أول")
        return

    df = pd.read_csv(INPUT_PATH, encoding="utf-8-sig")
    print(f"عدد الصفوف: {len(df)}")

    verdicts = df.apply(compute_final_verdict, axis=1)
    df["التوصية_النهائية"] = [v[0] for v in verdicts]
    df["سبب_التوصية_النهائية"] = [v[1] for v in verdicts]
    df["درجة_الثقة"] = df.apply(compute_confidence, axis=1)

    # ترتيب منطقي: كل مجموعة مترابطة مع بعض (التوصية، ثم الهوية، ثم مواصفات العقار،
    # ثم الإيجار، ثم العائد، ثم مقارنة السعر، ثم تحليل الـ LLM، والوصف الطويل آخر شي)
    ordered_cols = [
        # 1) التوصية النهائية (الأهم -- أول شي يشوفه القارئ)
        "التوصية_النهائية", "درجة_الثقة", "سبب_التوصية_النهائية",
        # 2) هوية العقار
        "listing_id", "url", "title", "district", "direction", "ضمن_الـ11_حي_التجريبية",
        # 3) مواصفات العقار (كلها مع بعض)
        "price", "area_sqm", "price_per_sqm", "rooms", "bathrooms", "livings",
        "age_years", "مؤثثة", "رقم_التواصل",
        # 4) الإيجار المتوقع (النطاق كامل متجاور)
        "rent_low", "rent_mid", "rent_high",
        # 5) العائد (النطاق كامل متجاور)
        "yield_low_pct", "yield_mid_pct", "yield_high_pct",
        "verdict_yield", "verdict_yield_reason",
        # 6) مقارنة السعر بالصفقات الرسمية (كلها مع بعض)
        "ad_price_per_sqm", "comparable_deals_count", "comparable_median_price_per_sqm",
        "ratio", "verdict_price", "verdict_price_reason",
        # 7) نقاط القوة والمخاطر
        "strengths", "risks",
        # 8) تحليل الـ LLM
        "is_multi_unit", "unit_label", "llm_corrections", "llm_notes",
        # 9) الوصف الطويل (آخر شي عشان ما يشتت باقي الأعمدة)
        "description",
    ]
    final_order = [c for c in ordered_cols if c in df.columns]
    # نضيف أي عمود إضافي ما ذكرناه (احتياطًا، عشان ما نفقد بيانات)
    final_order += [c for c in df.columns if c not in final_order]
    df = df[final_order]

    # نرتّب الصفوف: الفرص القوية أول، وداخل كل فئة نرتّب بالعائد الأعلى
    verdict_rank = {"🟢 فرصة قوية": 0, "🟡 تحتاج مراجعة": 1, "⚪ غير مكتمل": 2, "🔴 لا ينصح بها": 3}
    df["_rank"] = df["التوصية_النهائية"].map(verdict_rank).fillna(9)
    sort_cols = ["_rank"] + (["yield_low_pct"] if "yield_low_pct" in df.columns else [])
    df = df.sort_values(sort_cols, ascending=[True] + [False] * (len(sort_cols) - 1)).drop(columns=["_rank"])

    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print("\n--- توزيع التوصية النهائية ---")
    print(df["التوصية_النهائية"].value_counts().to_string())
    print("\n--- توزيع درجة الثقة ---")
    print(df["درجة_الثقة"].value_counts().to_string())
    print("\n--- التقاطع (التوصية × الثقة) ---")
    print(pd.crosstab(df["التوصية_النهائية"], df["درجة_الثقة"]).to_string())
    print(f"\nتم الحفظ: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
