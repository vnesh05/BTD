#!/usr/bin/env python3
"""
PDF Generation Executor
Simplified entry point to generate the project summary PDF.
Run: python3 run_pdf.py
"""

import sys
import subprocess
import os


def install_dependencies():
    """Install required dependencies."""
    try:
        import reportlab
        print("✓ reportlab is already installed")
        return True
    except ImportError:
        print("📦 Installing reportlab...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "reportlab>=4.0.0"])
            print("✓ reportlab installed successfully")
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to install reportlab")
            return False


def main():
    """Main execution function."""
    print("=" * 60)
    print("  Brain Tumour Detection - Project Summary PDF Generator")
    print("=" * 60)
    print()
    
    # Install dependencies
    print("Step 1: Checking dependencies...")
    if not install_dependencies():
        sys.exit(1)
    
    print()
    print("Step 2: Generating PDF...")
    print("-" * 60)
    
    # Import and run the PDF generator
    try:
        from generate_summary_pdf import create_summary_pdf
        pdf_path = create_summary_pdf("project_summary.pdf")
        
        # Verify file exists
        if os.path.isfile(pdf_path):
            file_size = os.path.getsize(pdf_path)
            print()
            print("=" * 60)
            print("✅ SUCCESS! PDF has been generated")
            print("-" * 60)
            print(f"📄 Filename:  project_summary.pdf")
            print(f"📊 File size: {file_size:,} bytes ({file_size / 1024:.1f} KB)")
            print(f"📍 Location:  {os.path.abspath(pdf_path)}")
            print("=" * 60)
            return 0
        else:
            print("❌ PDF file was not created")
            return 1
            
    except Exception as e:
        print(f"❌ Error during PDF generation: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
