#!/bin/bash

# PRefLexOR Presentation Site Builder

echo "🚀 PRefLexOR Presentation Site Builder"
echo "====================================="

# Function to check if Quarto is installed
check_quarto() {
    if ! command -v quarto &> /dev/null; then
        echo "❌ Quarto is not installed!"
        echo "Please install Quarto from: https://quarto.org/docs/get-started/"
        exit 1
    fi
    echo "✅ Quarto is installed: $(quarto --version)"
}

# Function to build the site
build_site() {
    echo "🔨 Building presentation site..."
    quarto render
    
    if [ $? -eq 0 ]; then
        echo "✅ Site built successfully!"
        echo "📁 Output directory: _site/"
    else
        echo "❌ Build failed!"
        exit 1
    fi
}

# Function to preview the site
preview_site() {
    echo "👀 Starting preview server..."
    echo "🌐 Site will be available at: http://localhost:4000"
    echo "Press Ctrl+C to stop the server"
    quarto preview
}

# Function to clean build artifacts
clean_site() {
    echo "🧹 Cleaning build artifacts..."
    rm -rf _site
    rm -rf .quarto
    echo "✅ Clean complete!"
}

# Function to publish to GitHub Pages
publish_github() {
    echo "📤 Publishing to GitHub Pages..."
    quarto publish gh-pages --no-prompt
}

# Function to test Mermaid
test_mermaid() {
    echo "🧪 Testing Mermaid diagrams..."
    quarto render test-mermaid.qmd
    
    if [ $? -eq 0 ]; then
        echo "✅ Mermaid test built successfully!"
        echo "🌐 Open _site/test-mermaid.html to check if diagrams render"
    else
        echo "❌ Mermaid test failed!"
    fi
}

# Function to fix common issues
fix_issues() {
    echo "🔧 Applying common fixes..."
    
    # Use simple configuration
    if [ -f "_quarto-simple.yml" ]; then
        echo "📝 Switching to simple configuration..."
        cp _quarto.yml _quarto-backup.yml
        cp _quarto-simple.yml _quarto.yml
        echo "✅ Configuration switched"
    fi
    
    # Clear cache
    echo "🧹 Clearing cache..."
    rm -rf .quarto _site
    
    # Rebuild
    echo "🔨 Rebuilding..."
    quarto render
    
    if [ $? -eq 0 ]; then
        echo "✅ Site rebuilt with fixes!"
    else
        echo "❌ Rebuild failed - check TROUBLESHOOTING.md"
    fi
}

# Main menu
show_menu() {
    echo ""
    echo "Select an option:"
    echo "1) Build site"
    echo "2) Preview site (development)"
    echo "3) Clean build artifacts"
    echo "4) Test Mermaid diagrams"
    echo "5) Fix common issues"
    echo "6) Publish to GitHub Pages"
    echo "7) Exit"
    echo ""
    read -p "Enter choice [1-7]: " choice
    
    case $choice in
        1)
            build_site
            show_menu
            ;;
        2)
            preview_site
            ;;
        3)
            clean_site
            show_menu
            ;;
        4)
            test_mermaid
            show_menu
            ;;
        5)
            fix_issues
            show_menu
            ;;
        6)
            publish_github
            show_menu
            ;;
        7)
            echo "👋 Goodbye!"
            exit 0
            ;;
        *)
            echo "❌ Invalid option!"
            show_menu
            ;;
    esac
}

# Check if command line argument is provided
if [ $# -eq 1 ]; then
    case $1 in
        build)
            check_quarto
            build_site
            ;;
        preview)
            check_quarto
            preview_site
            ;;
        clean)
            clean_site
            ;;
        test)
            check_quarto
            test_mermaid
            ;;
        fix)
            fix_issues
            ;;
        publish)
            check_quarto
            publish_github
            ;;
        *)
            echo "Usage: $0 {build|preview|clean|test|fix|publish}"
            exit 1
            ;;
    esac
else
    # Show interactive menu
    check_quarto
    show_menu
fi