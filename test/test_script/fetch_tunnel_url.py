"""Fetch tunnel URL from cpolar local webui."""
import re
import urllib.request

html = urllib.request.urlopen("http://127.0.0.1:4040/", timeout=5).read().decode("utf-8", "replace")
print("len:", len(html))
urls = re.findall(r"https?://[a-zA-Z0-9\-\.]+\.cpolar\.cn[^\s\"'<>]*", html)
print("cpolar urls found:", sorted(set(urls)) if urls else "none")
print("--- first 2000 chars ---")
print(html[:2000])
