#!/usr/bin/env python3
"""
Script to fix Mermaid diagram contrast issues in all QMD files
"""

import re
import os

# Standard theme configuration for good contrast
STANDARD_THEME = "%%{init: {'theme':'base', 'themeVariables': { 'primaryColor':'#ffffff', 'primaryTextColor':'#000000', 'primaryBorderColor':'#2e8b57', 'lineColor':'#2e8b57'}}}%%"

# Color mappings for better contrast
STYLE_REPLACEMENTS = {
    # Old problematic styles -> New high-contrast styles
    r"style\s+(\w+)\s+fill:#([0-9a-fA-F]{3,6})(?:,color:#fff)?": lambda m: f"style {m.group(1)} fill:#{get_light_color(m.group(2))},stroke:#{get_border_color(m.group(2))},stroke-width:2px,color:#000",
    r"style\s+(\w+)\s+fill:#([0-9a-fA-F]{3,6}),color:#([0-9a-fA-F]{3,6})": lambda m: f"style {m.group(1)} fill:#{get_light_color(m.group(2))},stroke:#{get_border_color(m.group(2))},stroke-width:2px,color:#000",
    r"style\s+(\w+)\s+fill:#([0-9a-fA-F]{3,6})$": lambda m: f"style {m.group(1)} fill:#{get_light_color(m.group(2))},stroke:#{get_border_color(m.group(2))},stroke-width:2px,color:#000",
}

def get_light_color(hex_color):
    """Convert dark colors to light equivalents"""
    color_map = {
        "2e8b57": "e7f3ff",  # green -> light blue
        "28a745": "e8f5e8",  # green -> light green
        "dc3545": "ffebee",  # red -> light red
        "ffc107": "fff3cd",  # yellow -> light yellow
        "007bff": "e7f3ff",  # blue -> light blue
        "6c757d": "f8f9fa",  # gray -> light gray
        "17a2b8": "d4edda",  # teal -> light green
        "fd7e14": "fff3cd",  # orange -> light yellow
    }
    
    # Handle 3-digit hex
    if len(hex_color) == 3:
        hex_color = "".join([c*2 for c in hex_color])
    
    return color_map.get(hex_color.lower(), "ffffff")  # default to white

def get_border_color(hex_color):
    """Get appropriate border color"""
    color_map = {
        "2e8b57": "2e8b57",  # green
        "28a745": "28a745",  # green
        "dc3545": "dc3545",  # red
        "ffc107": "ffc107",  # yellow
        "007bff": "007bff",  # blue
        "6c757d": "6c757d",  # gray
        "17a2b8": "28a745",  # teal -> green
        "fd7e14": "ffc107",  # orange -> yellow
    }
    
    # Handle 3-digit hex
    if len(hex_color) == 3:
        hex_color = "".join([c*2 for c in hex_color])
    
    return color_map.get(hex_color.lower(), "2e8b57")  # default to green

def fix_mermaid_init(content):
    """Replace problematic Mermaid init blocks with standard theme"""
    
    # Pattern to match Mermaid init blocks
    init_pattern = r"%%\{init:\s*\{[^}]*\}\s*\}%%"
    
    def replace_init(match):
        return STANDARD_THEME
    
    return re.sub(init_pattern, replace_init, content)

def fix_mermaid_styles(content):
    """Fix individual style statements for better contrast"""
    
    # Add standard styles for common patterns
    fixes = [
        # Remove problematic color combinations
        (r"style\s+(\w+)\s+fill:#([0-9a-fA-F]{6}),color:#fff", 
         r"style \1 fill:#ffffff,stroke:#\2,stroke-width:2px,color:#000"),
        
        # Fix dark fills without explicit text color
        (r"style\s+(\w+)\s+fill:#(2e8b57|28a745|dc3545|007bff|6c757d)(?![,])", 
         r"style \1 fill:#ffffff,stroke:#\2,stroke-width:2px,color:#000"),
        
        # Fix light fills that need borders
        (r"style\s+(\w+)\s+fill:#(fff|ffffff|f8f9fa|e7f3ff|e8f5e8|ffebee|fff3cd)(?![,])", 
         r"style \1 fill:#\2,stroke:#2e8b57,stroke-width:2px,color:#000"),
    ]
    
    for pattern, replacement in fixes:
        content = re.sub(pattern, replacement, content)
    
    return content

def process_file(filepath):
    """Process a single QMD file to fix Mermaid contrast issues"""
    
    print(f"Processing {filepath}...")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # Fix init blocks
    content = fix_mermaid_init(content)
    
    # Fix style statements
    content = fix_mermaid_styles(content)
    
    if content != original_content:
        # Backup original file
        backup_path = filepath + '.backup'
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.write(original_content)
        
        # Write fixed content
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"  ✅ Fixed {filepath} (backup saved to {backup_path})")
        return True
    else:
        print(f"  ℹ️  No changes needed for {filepath}")
        return False

def main():
    """Main function to process all QMD files"""
    
    qmd_files = [
        "technical-deep-dive.qmd",
        "visual-guide.qmd",
        "test-mermaid.qmd"
    ]
    
    print("🎨 Fixing Mermaid diagram contrast issues...")
    print("=" * 50)
    
    fixed_count = 0
    
    for qmd_file in qmd_files:
        if os.path.exists(qmd_file):
            if process_file(qmd_file):
                fixed_count += 1
        else:
            print(f"  ⚠️  File not found: {qmd_file}")
    
    print("=" * 50)
    print(f"✅ Processing complete! Fixed {fixed_count} files.")
    print("\nNote: Original files backed up with .backup extension")
    print("Test your changes with: ./build.sh test")

if __name__ == "__main__":
    main()