# Troubleshooting Guide for PRefLexOR Presentation Site

## Mermaid Diagrams Not Rendering

### Issue
Mermaid diagrams show in VS Code preview but don't render on the website.

### Solutions

#### Option 1: Use Simple Configuration
Replace `_quarto.yml` with `_quarto-simple.yml`:
```bash
mv _quarto.yml _quarto-backup.yml
mv _quarto-simple.yml _quarto.yml
```

#### Option 2: Test Mermaid Separately
1. Build the test page:
   ```bash
   quarto render test-mermaid.qmd
   ```
2. Open `_site/test-mermaid.html` in browser
3. If diagrams show here, the issue is configuration-specific

#### Option 3: Check Quarto Version
```bash
quarto --version
```
Ensure you have Quarto 1.4+ for best Mermaid support.

#### Option 4: Manual Mermaid Installation
If Quarto's built-in Mermaid isn't working:
```bash
# Install mermaid-cli globally
npm install -g @mermaid-js/mermaid-cli

# Convert diagrams to images
mmdc -i diagram.mmd -o diagram.png
```

#### Option 5: Use Alternative Diagram Formats

Replace Mermaid with SVG or PNG diagrams:
```markdown
![Architecture Diagram](images/architecture.svg)
```

### Debug Steps

1. **Check Browser Console**
   - Open Developer Tools (F12)
   - Look for JavaScript errors
   - Check if Mermaid library loads

2. **Validate Mermaid Syntax**
   - Test diagrams at https://mermaid.live/
   - Ensure no syntax errors

3. **Check Network Tab**
   - Verify Mermaid CDN loads
   - Check for 404 errors

4. **Try Different Browsers**
   - Test in Chrome, Firefox, Safari
   - Check if it's browser-specific

### Common Fixes

#### Fix 1: Remove Custom Mermaid Configuration
```yaml
format:
  html:
    # Remove custom mermaid configuration
    # Use only:
    mermaid:
      theme: default
```

#### Fix 2: Use PNG Format
```yaml
format:
  html:
    mermaid-format: png
```

#### Fix 3: Disable JavaScript Optimizations
```yaml
format:
  html:
    minimal: false
```

## Other Common Issues

### Build Errors

#### Missing Dependencies
```bash
# Reinstall Quarto
brew reinstall quarto

# Clear cache
rm -rf .quarto _site
```

#### Syntax Errors
```bash
# Check specific file
quarto check file.qmd

# Validate all files
quarto check
```

### Styling Issues

#### CSS Not Loading
1. Check `styles.css` path in `_quarto.yml`
2. Clear browser cache
3. Hard refresh (Ctrl+F5 or Cmd+Shift+R)

#### Responsive Issues
1. Test in mobile view
2. Check CSS media queries
3. Validate HTML structure

### Performance Issues

#### Slow Rendering
```yaml
execute:
  freeze: true  # Cache execution results
```

#### Large File Sizes
1. Optimize images
2. Use PNG for Mermaid instead of SVG
3. Enable compression

## Quick Fixes Script

Create `fix-site.sh`:
```bash
#!/bin/bash
echo "Fixing common issues..."

# Clear build artifacts
rm -rf _site .quarto

# Use simple configuration
if [ -f "_quarto-simple.yml" ]; then
    cp _quarto-simple.yml _quarto.yml
    echo "Using simple configuration"
fi

# Rebuild
quarto render

echo "Site rebuilt with fixes"
```

## Getting Help

1. **Check Quarto Documentation**: https://quarto.org/docs/authoring/diagrams.html
2. **Mermaid Documentation**: https://mermaid.js.org/
3. **Test Individual Components**: Use `test-mermaid.qmd`
4. **Community Support**: Quarto GitHub Discussions

## Version Information

Works best with:
- Quarto 1.4.550+
- Modern browsers (Chrome 90+, Firefox 88+, Safari 14+)
- Node.js 16+ (if using mermaid-cli)

## Final Resort: Static Images

If Mermaid continues to fail, convert to static images:
```bash
# Install mermaid-cli
npm install -g @mermaid-js/mermaid-cli

# Convert diagrams
mmdc -i flowchart.mmd -o flowchart.png -w 1024 -H 768

# Use in markdown
![Flowchart](flowchart.png)
```