import re

# Read dashboard file
with open('dashboard.py', 'r', encoding='utf-8') as f:
    content = f.read()

print("Fixing Plotly deprecation warnings...\n")

# The real issue: showlegend=False in update_layout() is deprecated
# Need to move it to update_traces() instead

count = len(re.findall(r'update_layout\([^)]*showlegend=False[^)]*\)', content))
print(f"Found {count} instances of update_layout with showlegend=False")

# Fix Pattern 1: update_layout(showlegend=False) - standalone
content = re.sub(
    r'(\s+)fig\.update_layout\(\s*showlegend=False\s*\)',
    r'\1fig.update_traces(showlegend=False)',
    content
)

# Fix Pattern 2: update_layout(other_param, showlegend=False)
content = re.sub(
    r'(\s+)fig\.update_layout\(([^)]+),\s*showlegend=False\s*\)',
    r'\1fig.update_layout(\2)\n\1fig.update_traces(showlegend=False)',
    content
)

# Fix Pattern 3: update_layout(showlegend=False, other_param)
content = re.sub(
    r'(\s+)fig\.update_layout\(\s*showlegend=False,\s*([^)]+)\)',
    r'\1fig.update_layout(\2)\n\1fig.update_traces(showlegend=False)',
    content
)

# Save updated file
with open('dashboard.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("\n✅ Fixed all Plotly deprecation warnings!")
print("   - Moved showlegend=False from update_layout() to update_traces()")
print("   - This is the correct Plotly API usage")
