#!/usr/bin/env python3
"""
Professional PDF Summary Generator
Generates a comprehensive project summary and onboarding document.
Uses reportlab for clean, professional formatting.
"""

from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, 
    PageBreak, Image, KeepTogether, Preformatted
)
from reportlab.lib import colors
from datetime import datetime


def create_summary_pdf(output_path="project_summary.pdf"):
    """Generate a professional PDF summary document."""
    
    # Create PDF document
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=0.75*inch,
        leftMargin=0.75*inch,
        topMargin=0.75*inch,
        bottomMargin=0.75*inch
    )
    
    # Custom styles
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1f4788'),
        spaceAfter=12,
        spaceBefore=0,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    heading1_style = ParagraphStyle(
        'CustomHeading1',
        parent=styles['Heading1'],
        fontSize=16,
        textColor=colors.HexColor('#2e5c8a'),
        spaceAfter=10,
        spaceBefore=12,
        fontName='Helvetica-Bold'
    )
    
    heading2_style = ParagraphStyle(
        'CustomHeading2',
        parent=styles['Heading2'],
        fontSize=13,
        textColor=colors.HexColor('#3d7ca8'),
        spaceAfter=8,
        spaceBefore=10,
        fontName='Helvetica-Bold'
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['BodyText'],
        fontSize=11,
        alignment=TA_JUSTIFY,
        spaceAfter=10,
        leading=14
    )
    
    bullet_style = ParagraphStyle(
        'CustomBullet',
        parent=styles['BodyText'],
        fontSize=11,
        leftIndent=20,
        spaceAfter=6,
        leading=13
    )
    
    code_style = ParagraphStyle(
        'Code',
        parent=styles['BodyText'],
        fontSize=9,
        fontName='Courier',
        textColor=colors.HexColor('#333333'),
        leftIndent=20,
        spaceAfter=8,
        leading=11
    )
    
    # Content
    content = []
    
    # Title section
    content.append(Spacer(1, 0.2*inch))
    content.append(Paragraph(
        "Brain Tumour Detection CNN",
        title_style
    ))
    content.append(Paragraph(
        "Comprehensive Executive Summary & Onboarding Guide",
        ParagraphStyle(
            'Subtitle',
            parent=styles['Normal'],
            fontSize=12,
            textColor=colors.HexColor('#555555'),
            alignment=TA_CENTER,
            spaceAfter=12
        )
    ))
    content.append(Spacer(1, 0.1*inch))
    
    # Metadata line
    content.append(Paragraph(
        f"<i>Repository: vnesh05/CNN-Brain-Tumour-Detection | Generated: {datetime.now().strftime('%B %d, %Y')}</i>",
        ParagraphStyle(
            'Meta',
            parent=styles['Normal'],
            fontSize=9,
            textColor=colors.HexColor('#888888'),
            alignment=TA_CENTER,
            spaceAfter=18
        )
    ))
    
    # ==================== SECTION 1: HIGH-LEVEL OVERVIEW ====================
    content.append(Paragraph("1. High-Level Overview", heading1_style))
    content.append(Paragraph(
        "This project implements a <b>Convolutional Neural Network (CNN) model</b> for automated brain "
        "tumour detection from MRI scan images using PyTorch and the Adam optimizer. It addresses the challenge "
        "of manual radiologist review by automating tumour classification to improve diagnostic speed and consistency. "
        "The target users are medical researchers, healthcare IT professionals, and practitioners exploring deep learning "
        "applications in medical imaging.",
        body_style
    ))
    content.append(Spacer(1, 0.1*inch))
    
    # ==================== SECTION 2: ARCHITECTURE & TECH STACK ====================
    content.append(Paragraph("2. Architecture &amp; Tech Stack", heading1_style))
    
    content.append(Paragraph("<b>Frameworks &amp; Libraries:</b>", heading2_style))
    frameworks = [
        "<b>PyTorch</b> – Deep learning framework for model definition and training",
        "<b>OpenCV (cv2)</b> – Image preprocessing, resizing, and augmentation",
        "<b>NumPy</b> – Array operations and tensor conversions",
        "<b>scikit-learn</b> – Evaluation metrics (accuracy, precision, recall, F1-score, confusion matrix)"
    ]
    for framework in frameworks:
        content.append(Paragraph(f"• {framework}", bullet_style))
    content.append(Spacer(1, 0.08*inch))
    
    content.append(Paragraph("<b>Core Model:</b>", heading2_style))
    content.append(Paragraph(
        "<b>DeeperCNN</b> – A three-layer convolutional architecture with max pooling, followed by fully "
        "connected layers with dropout (0.4) for regularization and sigmoid activation for binary classification.",
        bullet_style
    ))
    content.append(Spacer(1, 0.08*inch))
    
    content.append(Paragraph("<b>Directory Structure:</b>", heading2_style))
    dir_structure = """<font name="Courier" size="9">
.
├── model.py              Model definition (DeeperCNN class)
├── train.py              Training pipeline with early stopping
├── predict.py            Single-image inference script
├── test.py               Batch evaluation on test set
├── best_mri_model_v2.pth Pre-trained model weights (~5MB)
├── data/                 Dataset directory
│   ├── train/            Training images
│   ├── val/              Validation images
│   └── test/             Test images
├── test_images/          Sample images for inference
└── README.md             Project documentation
    </font>"""
    content.append(Paragraph(dir_structure, code_style))
    content.append(Spacer(1, 0.08*inch))
    
    content.append(Paragraph("<b>Data &amp; Execution Flow:</b>", heading2_style))
    flow_steps = [
        "<b>Preprocessing:</b> MRI images are read from data/, resized to 128×128, normalized (0–1 range), and converted to RGB",
        "<b>Augmentation:</b> Training images are randomly flipped horizontally and rotated (±15°) to reduce overfitting",
        "<b>Training:</b> DeeperCNN is trained using BCE loss and Adam optimizer (lr=0.0001) for up to 100 epochs with early stopping",
        "<b>Inference:</b> Preprocessed images pass through CNN layers and output a probability; predictions thresholded at 0.5"
    ]
    for i, step in enumerate(flow_steps, 1):
        content.append(Paragraph(f"{i}. {step}", bullet_style))
    content.append(Spacer(1, 0.1*inch))
    
    # ==================== SECTION 3: KEY FEATURES & HIGHLIGHTS ====================
    content.append(Paragraph("3. Key Features &amp; Highlights", heading1_style))
    
    content.append(Paragraph("<b>Core Functional Modules:</b>", heading2_style))
    modules = [
        "<b>model.py:</b> Defines CNN with 3 convolutional layers (16, 32, 64 filters) and dropout regularization",
        "<b>train.py:</b> Training loop with early stopping (patience=15), GPU/CPU auto-detection, data augmentation, and checkpoint saving",
        "<b>test.py:</b> Comprehensive evaluation using accuracy, classification report, and confusion matrix",
        "<b>predict.py:</b> Command-line inference tool for single-image predictions"
    ]
    for module in modules:
        content.append(Paragraph(f"• {module}", bullet_style))
    content.append(Spacer(1, 0.08*inch))
    
    content.append(Paragraph("<b>Standout Technical Implementations:</b>", heading2_style))
    features = [
        "<b>Binary Classification via Sigmoid + BCE Loss:</b> Appropriate for two-class medical imaging with probabilistic output",
        "<b>Dropout Regularization (0.4):</b> Reduces overfitting in fully connected layers",
        "<b>Early Stopping:</b> Prevents overtraining; saves best model checkpoints automatically",
        "<b>Data Augmentation:</b> Improves generalization with domain-appropriate transformations (rotation, flipping)"
    ]
    for feature in features:
        content.append(Paragraph(f"• {feature}", bullet_style))
    content.append(Spacer(1, 0.1*inch))
    
    # ==================== SECTION 4: SETUP & ENTRY POINTS ====================
    content.append(Paragraph("4. Setup &amp; Entry Points", heading1_style))
    
    content.append(Paragraph("<b>Key Entry Points:</b>", heading2_style))
    
    # Create table for entry points
    entry_table_data = [
        ["Script", "Purpose", "Command"],
        ["train.py", "Train the CNN from scratch", "python train.py"],
        ["test.py", "Evaluate model on test set", "python test.py"],
        ["predict.py", "Classify a single image", "python predict.py &lt;image_path&gt;"]
    ]
    entry_table = Table(entry_table_data, colWidths=[1.2*inch, 2.2*inch, 2.1*inch])
    entry_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2e5c8a')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f5f5')])
    ]))
    content.append(entry_table)
    content.append(Spacer(1, 0.12*inch))
    
    content.append(Paragraph("<b>Minimal Run Instructions:</b>", heading2_style))
    
    content.append(Paragraph("1. <b>Install dependencies:</b>", bullet_style))
    content.append(Preformatted("pip install torch torchvision opencv-python numpy scikit-learn", code_style))
    content.append(Spacer(1, 0.05*inch))
    
    content.append(Paragraph("2. <b>Prepare data structure</b> (assuming MRI dataset is available):", bullet_style))
    data_structure = """<font name="Courier" size="9">
data/
├── train/tumor/ ├── train/no_tumor/
├── val/tumor/   ├── val/no_tumor/
├── test/tumor/  └── test/no_tumor/
    </font>"""
    content.append(Paragraph(data_structure, code_style))
    content.append(Spacer(1, 0.05*inch))
    
    content.append(Paragraph("3. <b>Train the model:</b>", bullet_style))
    content.append(Preformatted("python train.py", code_style))
    content.append(Spacer(1, 0.03*inch))
    
    content.append(Paragraph("4. <b>Test the model:</b>", bullet_style))
    content.append(Preformatted("python test.py", code_style))
    content.append(Spacer(1, 0.03*inch))
    
    content.append(Paragraph("5. <b>Make predictions:</b>", bullet_style))
    content.append(Preformatted("python predict.py test_images/sample_mri.jpg", code_style))
    content.append(Spacer(1, 0.1*inch))
    
    # ==================== PAGE BREAK ====================
    content.append(PageBreak())
    
    # ==================== SECTION 5: PITCH & TALKING POINTS ====================
    content.append(Spacer(1, 0.1*inch))
    content.append(Paragraph("5. Pitch &amp; Talking Points", heading1_style))
    
    content.append(Paragraph("<b>30-Second Elevator Pitch:</b>", heading2_style))
    pitch_box_style = ParagraphStyle(
        'PitchBox',
        parent=styles['BodyText'],
        fontSize=11,
        alignment=TA_JUSTIFY,
        spaceAfter=12,
        leading=14,
        leftIndent=15,
        rightIndent=15,
        borderColor=colors.HexColor('#3d7ca8'),
        borderWidth=2,
        borderRadius=4,
        textColor=colors.HexColor('#1f4788')
    )
    pitch_text = (
        "This project demonstrates an end-to-end deep learning pipeline for medical image analysis. "
        "Using a custom CNN trained on MRI scans with the Adam optimizer, it automates brain tumour detection "
        "with binary classification. The model includes practical engineering—early stopping, data augmentation, "
        "GPU support, and comprehensive evaluation metrics—making it production-ready for proof-of-concept healthcare AI applications."
    )
    content.append(Paragraph(f"<i>\"{pitch_text}\"</i>", pitch_box_style))
    content.append(Spacer(1, 0.1*inch))
    
    content.append(Paragraph("<b>Interview &amp; Stakeholder Talking Points:</b>", heading2_style))
    talking_points = [
        (
            "🎯 <b>Clinical Impact:</b>",
            "Automates repetitive radiology review, reducing manual workload and enabling faster diagnosis—"
            "critical for time-sensitive conditions like brain tumours."
        ),
        (
            "🔧 <b>Production-Grade Design:</b>",
            "Implements early stopping to prevent overfitting, GPU acceleration for scalability, and "
            "checkpoint saving for model persistence—all essentials for real-world ML systems."
        ),
        (
            "📊 <b>Rigorous Evaluation:</b>",
            "Uses scikit-learn metrics (accuracy, precision, recall, F1, confusion matrix) rather than a single metric, "
            "following best practices for medical AI validation."
        ),
        (
            "🛠️ <b>Modular Architecture:</b>",
            "Separates concerns (model definition, training loop, inference, evaluation), making the code maintainable "
            "and extensible for future tasks like tumour localization or multi-class classification."
        )
    ]
    for title, desc in talking_points:
        content.append(Paragraph(f"• {title}", bullet_style))
        desc_style = ParagraphStyle(
            'PointDesc',
            parent=styles['BodyText'],
            fontSize=10,
            leftIndent=35,
            spaceAfter=10,
            leading=12,
            textColor=colors.HexColor('#333333')
        )
        content.append(Paragraph(desc, desc_style))
    
    content.append(Spacer(1, 0.15*inch))
    
    # ==================== FOOTER ====================
    footer_style = ParagraphStyle(
        'Footer',
        parent=styles['Normal'],
        fontSize=9,
        textColor=colors.HexColor('#999999'),
        alignment=TA_CENTER,
        spaceAfter=0
    )
    content.append(Paragraph(
        "—<br/><i>This document was automatically generated by generate_summary_pdf.py</i>",
        footer_style
    ))
    
    # Build PDF
    doc.build(content)
    print(f"✓ PDF successfully generated: {output_path}")
    return output_path


if __name__ == "__main__":
    create_summary_pdf("project_summary.pdf")
