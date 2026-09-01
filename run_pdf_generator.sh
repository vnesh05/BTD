#!/bin/bash
# PDF Generation Runner Script
# Installs dependencies and executes the PDF generator

echo "🚀 Starting PDF Summary Generation..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is not installed."
    exit 1
fi

echo "✓ Python 3 found"
echo ""

# Install reportlab if not already installed
echo "📦 Checking for reportlab..."
python3 -c "import reportlab" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "   Installing reportlab..."
    pip install reportlab>=4.0.0
    if [ $? -eq 0 ]; then
        echo "✓ reportlab installed successfully"
    else
        echo "❌ Failed to install reportlab"
        exit 1
    fi
else
    echo "✓ reportlab already installed"
fi

echo ""
echo "━━���━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📄 Generating project_summary.pdf..."
echo ""

# Run the PDF generator
python3 generate_summary_pdf.py

# Check if PDF was created
if [ -f "project_summary.pdf" ]; then
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "✅ Success! PDF generated: project_summary.pdf"
    echo "   File size: $(ls -lh project_summary.pdf | awk '{print $5}')"
    echo "   Location: $(pwd)/project_summary.pdf"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
else
    echo "❌ Error: PDF file was not created"
    exit 1
fi
