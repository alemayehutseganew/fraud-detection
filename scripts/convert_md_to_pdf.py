import sys
from pathlib import Path
from markdown import markdown
from xhtml2pdf import pisa


def link_callback(uri, rel):
    # Resolve relative URIs to absolute filesystem paths
    if uri.startswith(('http://', 'https://')):
        return uri
    base = Path(rel).parent
    candidate = (base / uri).resolve()
    return str(candidate)


def md_to_pdf(md_path: Path, pdf_path: Path) -> int:
    text = md_path.read_text(encoding='utf-8')
    html_body = markdown(text, extensions=['fenced_code', 'tables', 'toc'])
    html = f"""<!doctype html>
<html><head><meta charset='utf-8'><style>img{{max-width:100%;}}</style></head>
<body>{html_body}</body></html>"""
    with pdf_path.open('wb') as f:
        result = pisa.CreatePDF(src=html, dest=f, link_callback=lambda uri, rel: link_callback(uri, md_path))
    return result.err


def main():
    if len(sys.argv) < 3:
        print('Usage: convert_md_to_pdf.py <input.md> <output.pdf>')
        sys.exit(2)
    md = Path(sys.argv[1])
    pdf = Path(sys.argv[2])
    if not md.exists():
        print(f'Markdown file not found: {md}')
        sys.exit(3)
    pdf.parent.mkdir(parents=True, exist_ok=True)
    err = md_to_pdf(md, pdf)
    if err:
        print('PDF generation reported errors.')
        sys.exit(1)
    print(f'PDF written to {pdf}')


if __name__ == '__main__':
    main()
