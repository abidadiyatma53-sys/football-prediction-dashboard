import requests
import re

for url in [
    'https://understat.com/league/EPL',
    'https://understat.com/league/EPL/2023',
    'https://understat.com/main'
]:
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}
    r = requests.get(url, headers=headers)
    print(f"URL: {url}, status: {r.status_code}, len: {len(r.text)}")
    matches = re.findall(r"([a-zA-Z0-9_]+)\s*=\s*JSON\.parse\(", r.text)
    print(f"Vars in {url}:", matches)
