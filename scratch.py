import re

with open('manuscripts/ref.bib', 'r') as f:
    content = f.read()

entries = content.split('@')

for entry in entries[1:]:
    is_arxiv = False
    journal_match = re.search(r'journal\s*=\s*[{"]\s*arXiv.*?["}]', entry, re.IGNORECASE)
    if journal_match:
        is_arxiv = True
        
    has_booktitle = re.search(r'booktitle\s*=', entry, re.IGNORECASE)
    
    if is_arxiv and not has_booktitle:
        match_key = re.match(r'^[a-zA-Z]+\{([^,]+),', entry)
        if match_key:
            key = match_key.group(1).strip()
            title_match = re.search(r'title\s*=\s*[{"](.*?)["}]', entry, re.IGNORECASE | re.DOTALL)
            title = title_match.group(1).replace('\n', ' ').strip() if title_match else 'No Title'
            print(f"Key: {key}\nTitle: {title}\n")
