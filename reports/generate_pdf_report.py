import os
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, HRFlowable, KeepTogether, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_PDF = BASE_DIR / "reports" / "final_report.pdf"
CHART_PATH = BASE_DIR / "scorer_results" / "candidate_december.png"


def create_pdf_report():
    OUTPUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUTPUT_PDF),
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()

    # Custom Color Palette
    PRIMARY = colors.HexColor("#064A56")
    SECONDARY = colors.HexColor("#2E8B57")
    DARK_TEXT = colors.HexColor("#1F2937")
    LIGHT_BG = colors.HexColor("#F8FAFC")
    BORDER_COLOR = colors.HexColor("#CBD5E1")
    ACCENT = colors.HexColor("#0284C7")

    # Typography Styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=28,
        textColor=PRIMARY,
        spaceAfter=12
    )

    heading1_style = ParagraphStyle(
        'SectionHeading1',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=15,
        leading=18,
        textColor=PRIMARY,
        spaceBefore=14,
        spaceAfter=8,
        keepWithNext=True
    )

    heading2_style = ParagraphStyle(
        'SectionHeading2',
        parent=styles['Heading3'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=15,
        textColor=SECONDARY,
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['BodyText'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        textColor=DARK_TEXT,
        spaceAfter=8
    )

    bullet_style = ParagraphStyle(
        'BulletDark',
        parent=body_style,
        leftIndent=15,
        firstLineIndent=-10,
        spaceAfter=4
    )

    callout_style = ParagraphStyle(
        'CalloutText',
        parent=body_style,
        fontName='Helvetica-Oblique',
        fontSize=9.5,
        leading=13.5,
        textColor=PRIMARY
    )

    story = []

    # Document Header
    story.append(Paragraph("Spotter AI — Freight Rate Prediction Assessment", title_style))
    story.append(HRFlowable(width="100%", thickness=2, color=PRIMARY, spaceBefore=0, spaceAfter=12))

    # 1. Executive Summary
    story.append(Paragraph("1. Executive Summary", heading1_style))
    exec_summary_text = (
        "This technical report presents an industry-grade Machine Learning solution for Spotter AI's Freight Rate "
        "Prediction challenge. The goal is to accurately forecast spot market rates (<code>posted_rate</code> in $) for freight loads "
        "based on origin/destination locations, load weight, equipment type, distance, and macro-market indicators.<br/><br/>"
        "<b>Key Accomplishments:</b><br/>"
        "• <b>Out-Of-Time Validation Rigor:</b> Designed a strict temporal out-of-time (OOT) validation framework (Jan–Aug train vs Sept–Oct val) "
        "mirroring the production evaluation setup (Jan–Oct development vs Nov–Dec deployment).<br/>"
        "• <b>Zero-Shot Geographic Generalization:</b> Implemented Haversine distance, coordinate deltas, and spatial midpoints to effectively "
        "handle unseen cities in the validation dataset.<br/>"
        "• <b>Model Architecture Excellence:</b> Built a high-performance <code>FreightEnsembleModel</code> blending <b>CatBoost</b>, <b>HistGradientBoosting</b>, "
        "and <b>LightGBM</b> trained on log-transformed rates. Achieved a validation <b>MAE of $127.09</b>, <b>MAPE of 5.45%</b>, <b>MedianAE of $49.99</b>, "
        "and an <b>R² of 0.8244</b>.<br/>"
        "• <b>Verification:</b> Passed 100% of official <code>score.py</code> automated validation checks for all 12,000 validation loads and 31 December chart predictions."
    )
    story.append(Paragraph(exec_summary_text, body_style))
    story.append(Spacer(1, 8))

    # 2. Data Understanding & Quality Audit
    story.append(Paragraph("2. Data Understanding & Quality Audit", heading1_style))
    data_audit_text = (
        "A rigorous data audit was conducted across all provided datasets:<br/>"
        "• <b>Development Set (<code>train-test.csv</code>):</b> 48,000 loads spanning 2025-01-01 to 2025-10-31.<br/>"
        "• <b>Validation Set (<code>validation.csv</code>):</b> 12,000 loads spanning 2025-11-01 to 2025-12-31.<br/>"
        "• <b>December Input Set (<code>december_chart_inputs.csv</code>):</b> 31 fixed daily loads for Lexington → Fort Wayne.<br/>"
        "• <b>Missing Values:</b> <code>weight</code> had 300 missing values in train (165 in val); <code>market_index</code> had 374 missing in train (249 in val). "
        "These were imputed using equipment-specific medians and global date medians without data leakage.<br/>"
        "• <b>Unseen Cities:</b> 8 pickup/delivery cities in the validation set do not appear in training data. Incorporating spatial coordinates ("
        "<code>pickup_lat</code>, <code>pickup_lon</code>, <code>delivery_lat</code>, <code>delivery_lon</code>) prevented failure on unseen locations."
    )
    story.append(Paragraph(data_audit_text, body_style))
    story.append(Spacer(1, 8))

    # Target Statistics Table
    target_table_data = [
        ["Metric", "Value ($)", "Interpretation"],
        ["Mean Posted Rate", "$2,373.98", "Average freight transaction rate"],
        ["Median Posted Rate", "$2,030.76", "Robust measure of central tendency"],
        ["Min / Max Rate", "$57.22 / $25,533.00", "Wide price spread across short vs long hauls"],
        ["Target Skewness", "1.90", "Right-skewed target; log1p transformation used"]
    ]
    t1 = Table(target_table_data, colWidths=[120, 110, 310])
    t1.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), PRIMARY),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 8.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('GRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [LIGHT_BG, colors.white]),
    ]))
    story.append(t1)
    story.append(Spacer(1, 12))

    # 3. Feature Engineering & Spatial Generalization
    story.append(Paragraph("3. Feature Engineering Strategy", heading1_style))
    fe_text = (
        "To capture non-linear market relationships while avoiding data leakage, domain-inspired features were constructed:<br/>"
        "• <b>Haversine Distance & Route Circuity:</b> Great Circle distance computed from lat/lon coordinates. Route circuity ratio (<code>distance / (haversine + 1)</code>) "
        "captures highway detour factors (negatively correlated with rate per mile, r = -0.20).<br/>"
        "• <b>Spatial Deltas & Midpoints:</b> <code>delta_lat</code>, <code>delta_lon</code>, and regional midpoints model directional freight lanes.<br/>"
        "• <b>Payload Density:</b> <code>weight_per_mile</code> and log-scaled payload metrics.<br/>"
        "• <b>Market & Demand Interactions:</b> Interaction terms <code>distance * market_index</code> and <code>market_index * quote_signal</code>.<br/>"
        "• <b>Cyclical Calendar Encodings:</b> Sine/cosine transformations of <code>dayofweek</code> and <code>dayofyear</code> model weekly and annual seasonality."
    )
    story.append(Paragraph(fe_text, body_style))
    story.append(Spacer(1, 10))

    # 4. Out-Of-Time Validation & Model Benchmarking
    story.append(Paragraph("4. Validation Strategy & Model Benchmarking", heading1_style))
    val_strat_text = (
        "<b>Out-Of-Time (OOT) Split:</b> The development dataset (Jan–Oct 2025) was split temporally. Models were trained on "
        "Jan–Aug 2025 (38,477 loads) and evaluated on Sept–Oct 2025 (9,523 loads). This mimics the Nov–Dec 2025 validation deployment.<br/>"
        "<b>Model Comparison:</b> Evaluated 7 candidate model architectures using identical features and OOT splits."
    )
    story.append(Paragraph(val_strat_text, body_style))
    story.append(Spacer(1, 6))

    # Model Benchmarking Table
    model_table_data = [
        ["Model Architecture", "Target", "MAE ($)", "RMSE ($)", "MAPE (%)", "MedAE ($)", "R² Score"],
        ["Ridge Baseline", "Raw", "$194.43", "$659.45", "10.40%", "$120.43", "0.8133"],
        ["Random Forest", "Raw", "$184.82", "$680.07", "8.19%", "$64.07", "0.8014"],
        ["Extra Trees", "Raw", "$169.81", "$663.04", "7.44%", "$60.53", "0.8112"],
        ["XGBoost Regressor", "Log1p", "$153.91", "$665.55", "6.82%", "$57.16", "0.8098"],
        ["LightGBM Regressor", "Log1p", "$139.20", "$642.29", "6.04%", "$59.92", "0.8229"],
        ["HistGradientBoosting", "Log1p", "$128.03", "$639.58", "5.49%", "$48.33", "0.8243"],
        ["CatBoost Regressor", "Log1p", "$127.99", "$639.31", "5.70%", "$52.91", "0.8245"],
        ["FreightEnsembleModel (Final)", "Log1p", "$127.09", "$639.54", "5.45%", "$49.99", "0.8244"]
    ]
    t2 = Table(model_table_data, colWidths=[140, 50, 65, 65, 65, 65, 60])
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), PRIMARY),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('GRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('ROWBACKGROUNDS', (0,1), (-1,-2), [LIGHT_BG, colors.white]),
        ('BACKGROUND', (0,-1), (-1,-1), colors.HexColor("#E0F2FE")),
        ('FONTNAME', (0,-1), (-1,-1), 'Helvetica-Bold'),
        ('TEXTCOLOR', (0,-1), (-1,-1), PRIMARY),
    ]))
    story.append(t2)
    story.append(Spacer(1, 12))

    # 5. Performance Breakdown & Error Analysis
    story.append(Paragraph("5. Detailed Performance Breakdown & Error Analysis", heading1_style))
    story.append(Paragraph("Performance by Equipment Type:", heading2_style))
    
    eq_table_data = [
        ["Equipment Type", "Load Count", "MAE ($)", "RMSE ($)", "MAPE (%)", "MedAE ($)", "R² Score"],
        ["Dry Van", "5,360", "$118.89", "$659.89", "5.04%", "$42.94", "0.7992"],
        ["Flatbed", "1,770", "$129.67", "$514.86", "5.93%", "$61.55", "0.8847"],
        ["Reefer (Refrigerated)", "2,393", "$143.55", "$675.46", "5.98%", "$62.65", "0.8269"]
    ]
    t3 = Table(eq_table_data, colWidths=[130, 70, 70, 70, 70, 70, 60])
    t3.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), SECONDARY),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('GRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [LIGHT_BG, colors.white]),
    ]))
    story.append(t3)
    story.append(Spacer(1, 8))

    story.append(Paragraph("Performance by Distance Tier:", heading2_style))
    dist_table_data = [
        ["Distance Tier", "Load Count", "MAE ($)", "RMSE ($)", "MAPE (%)", "MedAE ($)"],
        ["Short Haul (<500 mi)", "2,090", "$45.16", "$238.44", "5.50%", "$17.63"],
        ["Medium Haul (500–1,200 mi)", "3,836", "$96.97", "$433.28", "5.76%", "$51.51"],
        ["Long Haul (>1,200 mi)", "3,597", "$206.82", "$921.75", "5.08%", "$95.65"]
    ]
    t4 = Table(dist_table_data, colWidths=[150, 75, 75, 80, 80, 80])
    t4.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), SECONDARY),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('GRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [LIGHT_BG, colors.white]),
    ]))
    story.append(t4)
    story.append(Spacer(1, 12))

    # 6. Official December Prediction Chart
    story.append(Paragraph("6. Official December 2025 Prediction Chart", heading1_style))
    dec_chart_desc = (
        "The plot below illustrates the predicted spot rates generated by <code>FreightEnsembleModel</code> for 31 consecutive days "
        "in December 2025 for a fixed lane (Lexington to Fort Wayne, 360 miles, Dry Van, 32,000 lbs). The predictions exhibit "
        "realistic weekly cyclical variations, peaking during mid-week shipping spikes ($840–$843) and softening over weekend periods ($813–$818)."
    )
    story.append(Paragraph(dec_chart_desc, body_style))
    story.append(Spacer(1, 6))

    if CHART_PATH.exists():
        img = Image(str(CHART_PATH), width=520, height=230)
        story.append(img)
        story.append(Spacer(1, 10))

    # 7. Code Architecture & Submission Checklist
    story.append(Paragraph("7. Modular Code Architecture & Reproducibility", heading1_style))
    code_arch_text = (
        "The project is structured as a production-ready Python package in <code>src/</code>:<br/>"
        "• <code>src/config.py</code>: Centralized configurations, paths, and random seeds.<br/>"
        "• <code>src/data_preparation.py</code>: Clean data loading, imputation, and December input enrichment.<br/>"
        "• <code>src/features.py</code>: Leakage-free feature engineering transformer (fitted only on training data).<br/>"
        "• <code>src/model.py</code>: <code>FreightEnsembleModel</code> class definition.<br/>"
        "• <code>src/train.py</code>: Out-Of-Time evaluation and full retraining pipeline.<br/>"
        "• <code>src/predict.py</code>: Inference script producing <code>validation_predictions.csv</code> and updating <code>december_chart_inputs.csv</code>.<br/>"
        "• <code>src/evaluate.py</code>: Evaluation suite for calculating MAE, RMSE, MAPE, MedAE, R², and group breakdowns."
    )
    story.append(Paragraph(code_arch_text, body_style))
    story.append(Spacer(1, 12))

    doc.build(story)
    print(f"Generated updated PDF report successfully at {OUTPUT_PDF}")


if __name__ == "__main__":
    create_pdf_report()
