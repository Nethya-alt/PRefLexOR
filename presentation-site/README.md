# PRefLexOR Presentation Site

This directory contains the Quarto-based presentation website for PRefLexOR, showcasing transparent AI reasoning across six business applications.

## Prerequisites

1. **Install Quarto**
   ```bash
   # macOS
   brew install quarto
   
   # Windows
   # Download from https://quarto.org/docs/get-started/
   
   # Linux
   wget https://github.com/quarto-dev/quarto-cli/releases/download/v1.4.550/quarto-1.4.550-linux-amd64.deb
   sudo dpkg -i quarto-1.4.550-linux-amd64.deb
   ```

2. **Install Dependencies** (optional for enhanced features)
   ```bash
   pip install jupyter matplotlib plotly
   ```

## Building the Site

### Preview (Development)
```bash
# Navigate to presentation-site directory
cd presentation-site

# Preview with live reload
quarto preview

# The site will open at http://localhost:4000
```

### Build (Production)
```bash
# Build the static site
quarto render

# Output will be in _site directory
```

## Project Structure

```
presentation-site/
├── _quarto.yml           # Site configuration
├── index.qmd            # Landing page
├── executive-summary.qmd # Executive overview
├── technical-deep-dive.qmd # Technical documentation
├── visual-guide.qmd     # Implementation examples
├── styles.css           # Custom styling
├── _site/              # Generated site (after build)
└── README.md           # This file
```

## Key Features

- **Mermaid Diagrams**: Interactive flowcharts and visualizations
- **Code Highlighting**: Syntax highlighting with line numbers
- **Responsive Design**: Mobile-friendly layout
- **Dark Mode**: Toggle between light and dark themes
- **Search**: Built-in search functionality
- **Navigation**: Sidebar and top navigation

## Customization

### Modify Theme
Edit `_quarto.yml`:
```yaml
format:
  html:
    theme: 
      light: cosmo  # Change to: flatly, journal, etc.
      dark: darkly  # Change to: solar, superhero, etc.
```

### Add New Pages
1. Create new `.qmd` file
2. Add to `_quarto.yml` navigation:
```yaml
navbar:
  left:
    - href: your-new-page.qmd
      text: Your Page Title
```

### Custom Mermaid Themes
Mermaid diagrams use custom theming defined in `_quarto.yml`. Modify the `themeVariables` to change colors.

## Deployment Options

### GitHub Pages
```bash
# Configure for GitHub Pages
quarto publish gh-pages

# Follow the prompts
```

### Netlify
```bash
# Build site
quarto render

# Deploy _site directory to Netlify
# Via drag-and-drop or CLI
```

### Custom Server
```bash
# Build site
quarto render

# Copy _site contents to your web server
rsync -avz _site/ user@server:/var/www/html/
```

## Development Tips

1. **Live Preview**: Use `quarto preview` for development
2. **Check Links**: `quarto check` validates internal links
3. **Clean Build**: `rm -rf _site && quarto render` for fresh build
4. **Debug**: Add `echo: true` to code blocks for debugging

## Troubleshooting

### Mermaid Diagrams Not Rendering
- Ensure JavaScript is enabled
- Check browser console for errors
- Try clearing browser cache

### Build Errors
- Run `quarto check` to validate setup
- Ensure all .qmd files have proper YAML headers
- Check for syntax errors in Mermaid diagrams

### Styling Issues
- Clear browser cache
- Check `styles.css` is properly linked
- Verify no CSS conflicts

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with `quarto preview`
5. Submit a pull request

## License

This presentation is part of the PRefLexOR project. See main project LICENSE for details.

---

For more information about PRefLexOR, visit the [main repository](https://github.com/apingali/PRefLexOR).