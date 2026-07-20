"""
Convert the quantum report markdown to HTML, then use Word COM to save as .docx.
Pure stdlib — no external packages required.
"""
import re
import sys
import os

MD_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "outputs", "draft", "quantum_report_draft_v2.md"
)
HTML_PATH = MD_PATH.replace(".md", ".html")
DOCX_PATH = MD_PATH.replace(".md", ".docx")


def md_to_html(text: str) -> str:
    """Convert basic markdown to styled HTML."""

    # Escape HTML entities in text content first (but not inside code blocks)
    # We'll handle code blocks separately

    # Store code blocks
    code_blocks = []
    def store_code(m):
        code_blocks.append(m.group(0))
        return f"%%CODEBLOCK{len(code_blocks)-1}%%"
    text = re.sub(r'```.*?```', store_code, text, flags=re.DOTALL)

    # Store inline code
    inline_codes = []
    def store_inline(m):
        inline_codes.append(m.group(1))
        return f"%%INLINECODE{len(inline_codes)-1}%%"
    text = re.sub(r'`([^`]+)`', store_inline, text)

    # Store image references
    images = []
    def store_img(m):
        images.append((m.group(1), m.group(2)))
        return f"%%IMAGE{len(images)-1}%%"
    text = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', store_img, text)

    # Store links
    links = []
    def store_link(m):
        links.append((m.group(1), m.group(2)))
        return f"%%LINK{len(links)-1}%%"
    # Only match non-image links (not preceded by !)
    text = re.sub(r'(?<!!)\[([^\]]+)\]\(([^)]+)\)', store_link, text)

    # Footnotes
    footnotes = {}
    def store_footnote(m):
        key = m.group(1)
        content = m.group(2)
        footnotes[key] = content
        return f'<sup><a href="#fn{key}" id="fnref{key}">[{key}]</a></sup>'
    text = re.sub(r'\[\^(\d+)\]:\s*(.+?)(?=\n\[\^|\n---|\Z)', store_footnote, text, flags=re.DOTALL)
    text = re.sub(r'\[\^(\d+)\]', r'<sup><a href="#fn\1" id="fnref\1">[\1]</a></sup>', text)

    # Headers (must be before horizontal rules)
    text = re.sub(r'^#### (.+)$', r'<h4>\1</h4>', text, flags=re.MULTILINE)
    text = re.sub(r'^### (.+)$', r'<h3>\1</h3>', text, flags=re.MULTILINE)
    text = re.sub(r'^## (.+)$', r'<h2>\1</h2>', text, flags=re.MULTILINE)
    text = re.sub(r'^# (.+)$', r'<h1>\1</h1>', text, flags=re.MULTILINE)

    # Horizontal rules
    text = re.sub(r'^---+$', r'<hr>', text, flags=re.MULTILINE)

    # Bold and italic
    text = re.sub(r'\*\*\*(.+?)\*\*\*', r'<strong><em>\1</em></strong>', text)
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'\*(.+?)\*', r'<em>\1</em>', text)

    # Blockquotes
    lines = text.split('\n')
    in_bq = False
    result_lines = []
    for line in lines:
        if line.startswith('> '):
            if not in_bq:
                result_lines.append('<blockquote>')
                in_bq = True
            result_lines.append(line[2:])
        else:
            if in_bq:
                result_lines.append('</blockquote>')
                in_bq = False
            result_lines.append(line)
    if in_bq:
        result_lines.append('</blockquote>')
    text = '\n'.join(result_lines)

    # Tables — find markdown table blocks and convert to HTML
    def convert_table(md_table: str) -> str:
        rows = [r for r in md_table.strip().split('\n') if r.strip() and not re.match(r'^[\|\s\-:]+$', r)]
        if len(rows) < 1:
            return md_table
        html = '<table>\n'
        for i, row in enumerate(rows):
            tag = 'th' if i == 0 else 'td'
            cells = [c.strip() for c in row.split('|') if c.strip() != '']
            html += '<tr>\n'
            for cell in cells:
                # Parse alignment from the cell content
                html += f'<{tag}>{cell}</{tag}>\n'
            html += '</tr>\n'
        html += '</table>\n'
        return html

    # Find table blocks (lines containing |)
    text_lines = text.split('\n')
    out_lines = []
    table_buffer = []
    in_table = False
    for line in text_lines:
        if '|' in line and line.strip().startswith('|'):
            table_buffer.append(line)
            in_table = True
        else:
            if in_table:
                out_lines.append(convert_table('\n'.join(table_buffer)))
                table_buffer = []
                in_table = False
            out_lines.append(line)
    if in_table:
        out_lines.append(convert_table('\n'.join(table_buffer)))
    text = '\n'.join(out_lines)

    # Paragraphs — wrap non-tag lines in <p>
    text_lines = text.split('\n')
    out_lines = []
    para_buffer = []
    for line in text_lines:
        if (line.strip().startswith('<') and not line.strip().startswith('<')):
            # This is a block-level tag already
            pass
        if re.match(r'^\s*<(h[1-6]|table|hr|blockquote|/blockquote|ul|ol|li|/ul|/ol|tr|/tr)', line):
            if para_buffer:
                out_lines.append('<p>' + '<br>'.join(para_buffer) + '</p>')
                para_buffer = []
            out_lines.append(line)
        elif line.strip() == '' or line.strip().startswith('<!--'):
            if para_buffer:
                out_lines.append('<p>' + '<br>'.join(para_buffer) + '</p>')
                para_buffer = []
            out_lines.append(line)
        else:
            para_buffer.append(line)
    if para_buffer:
        out_lines.append('<p>' + '<br>'.join(para_buffer) + '</p>')
    text = '\n'.join(out_lines)

    # Restore inline code
    for i, code in enumerate(inline_codes):
        text = text.replace(f"%%INLINECODE{i}%%", f'<code>{code}</code>')

    # Restore code blocks
    for i, block in enumerate(code_blocks):
        content = block.replace('```', '')
        # Remove language specifier if present
        content = re.sub(r'^[a-z]*\n', '', content)
        text = text.replace(f"%%CODEBLOCK{i}%%", f'<pre><code>{content.strip()}</code></pre>')

    # Restore images
    for i, (alt, src) in enumerate(images):
        text = text.replace(f"%%IMAGE{i}%%",
            f'<p class="figure"><em>{alt}</em></p>' if alt else '')

    # Restore links
    for i, (label, url) in enumerate(links):
        text = text.replace(f"%%LINK{i}%%", f'<a href="{url}">{label}</a>')

    return text


def main():
    # Read markdown
    with open(MD_PATH, 'r', encoding='utf-8') as f:
        md_content = f.read()

    # Convert to HTML body
    body_html = md_to_html(md_content)

    # Wrap in full HTML document with Chinese-friendly CSS
    full_html = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<style>
  @page {{ size: A4; margin: 2.54cm 3.18cm 2.54cm 3.18cm; }}
  body {{
    font-family: "SimSun", "宋体", serif;
    font-size: 12pt;
    line-height: 1.8;
    color: #000;
  }}
  h1 {{
    font-family: "SimHei", "黑体", sans-serif;
    font-size: 16pt;
    text-align: center;
    margin-top: 24pt;
    margin-bottom: 12pt;
  }}
  h2 {{
    font-family: "SimHei", "黑体", sans-serif;
    font-size: 14pt;
    margin-top: 18pt;
    margin-bottom: 9pt;
  }}
  h3 {{
    font-family: "SimHei", "黑体", sans-serif;
    font-size: 13pt;
    margin-top: 14pt;
    margin-bottom: 7pt;
  }}
  h4 {{
    font-family: "SimHei", "黑体", sans-serif;
    font-size: 12pt;
    margin-top: 12pt;
    margin-bottom: 6pt;
  }}
  p {{
    text-indent: 2em;
    margin: 6pt 0;
  }}
  p.figure {{
    text-indent: 0;
    text-align: center;
    font-size: 10pt;
    color: #555;
  }}
  blockquote {{
    margin: 8pt 0.5cm;
    padding: 6pt 10pt;
    border-left: 3pt solid #999;
    background: #f5f5f5;
    font-size: 11pt;
  }}
  blockquote p {{
    text-indent: 0;
  }}
  table {{
    border-collapse: collapse;
    width: 100%;
    margin: 12pt 0;
    font-size: 10pt;
  }}
  th, td {{
    border: 0.5pt solid #000;
    padding: 3pt 6pt;
    text-align: left;
  }}
  th {{
    background-color: #e0e0e0;
    font-weight: bold;
    text-align: center;
  }}
  td {{
    text-align: right;
  }}
  td:first-child {{
    text-align: left;
  }}
  p:has(+ table) {{
    text-indent: 0;
    font-size: 10pt;
    margin-bottom: 0;
  }}
  hr {{
    border: none;
    border-top: 0.5pt solid #999;
    margin: 18pt 0;
  }}
  code {{
    font-family: "Courier New", monospace;
    font-size: 10pt;
    background: #f0f0f0;
    padding: 1pt 3pt;
  }}
  pre {{
    font-family: "Courier New", monospace;
    font-size: 10pt;
    background: #f0f0f0;
    padding: 8pt;
    border: 0.5pt solid #ccc;
    overflow-x: auto;
  }}
  sup a {{
    font-size: 8pt;
    text-decoration: none;
  }}
  a {{
    color: #000;
    text-decoration: underline;
  }}
</style>
</head>
<body>
{body_html}
</body>
</html>'''

    # Write HTML
    with open(HTML_PATH, 'w', encoding='utf-8') as f:
        f.write(full_html)
    print(f"HTML saved to: {HTML_PATH}")

    print("Now use Word COM to convert HTML to DOCX...")
    print(f"Open the HTML in Word and Save As: {DOCX_PATH}")
    print("Or run: powershell -Command \"$w=New-Object -ComObject Word.Application; $d=$w.Documents.Open('...'); $d.SaveAs2('...', 16); $d.Close(); $w.Quit()\"")


if __name__ == "__main__":
    main()
