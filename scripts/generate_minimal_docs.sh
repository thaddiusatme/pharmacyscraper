#!/bin/bash
set -e

# Create directories if they don't exist
mkdir -p docs/_build/html

# Generate the minimal HTML documentation
cat > docs/_build/html/index.html << 'EOL'
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Pharmacy Scraper Documentation</title>
    <style>
        body { 
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            line-height: 1.6; 
            margin: 0; 
            padding: 20px;
            color: #24292e;
            max-width: 800px;
            margin: 0 auto;
        }
        .container { 
            padding: 20px; 
            border: 1px solid #e1e4e8; 
            border-radius: 6px; 
            margin: 20px 0;
        }
        h1 { 
            color: #2c3e50;
            border-bottom: 1px solid #eaecef;
            padding-bottom: 10px;
        }
        pre { 
            background: #f6f8fa; 
            padding: 16px; 
            border-radius: 6px; 
            overflow-x: auto;
            font-family: SFMono-Regular, Consolas, "Liberation Mono", Menlo, monospace;
        }
        .success { color: #22863a; }
        .error { color: #cb2431; }
        .warning { color: #b08800; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Pharmacy Scraper Documentation</h1>
        <p>Welcome to the Pharmacy Scraper documentation. This is a minimal documentation page generated by the CI system.</p>
        
        <div class="success">
            <h2>Documentation Status: Success</h2>
            <p>The documentation was generated successfully, but no custom documentation was found.</p>
        </div>
        
        <h2>Project Structure</h2>
        <pre>$(find . -type d | sort | sed 's|^./||' | grep -v "\.git" | grep -v "__pycache__" | sed 's|[^/]*/|   |g')</pre>
        
        <h2>Next Steps</h2>
        <p>To build the full documentation locally, run:</p>
        <pre>pip install sphinx sphinx-rtd-theme
cd docs
make html</pre>
    </div>
</body>
</html>
EOL

echo "Generated minimal documentation at docs/_build/html/index.html"
cat docs/_build/html/index.html
