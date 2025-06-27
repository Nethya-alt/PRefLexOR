#!/usr/bin/env python3
"""
Launch script for Legal Document Analysis Application
"""

import subprocess
import sys
import os

def main():
    """Launch the Legal Document Analysis application"""
    
    print("⚖️ Starting Legal Document Analysis Application...")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("app.py"):
        print("❌ Error: app.py not found. Please run from the legal_analysis directory.")
        sys.exit(1)
    
    # Install requirements if needed
    if os.path.exists("requirements.txt"):
        print("📦 Installing requirements...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                         check=True, capture_output=True)
            print("✅ Requirements installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"⚠️ Warning: Failed to install requirements: {e}")
    
    # Launch Streamlit
    print("\n🚀 Launching Legal Document Analysis...")
    print("📱 The application will open in your browser")
    print("🛑 Press Ctrl+C to stop the application")
    print("\n" + "=" * 50)
    
    try:
        subprocess.run([
            "streamlit", "run", "app.py",
            "--server.port", "8504",
            "--server.headless", "true"
        ])
    except KeyboardInterrupt:
        print("\n\n🛑 Application stopped by user")
    except FileNotFoundError:
        print("❌ Error: Streamlit not found. Please install with: pip install streamlit")
        sys.exit(1)

if __name__ == "__main__":
    main()