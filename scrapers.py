import json
import re
import time
from dataclasses import dataclass, field
from urllib.parse import urlparse, urljoin

import requests
from bs4 import BeautifulSoup


@dataclass
class FetchResult:
    source_name: str
    url: str
    success: bool
    cleaned_text: str
    error_message: str
    pages_fetched: int = 1


DOMAIN_NAMES = {
    "guide.michelin.com": "Michelin Guide",
    "theinfatuation.com": "The Infatuation",
    "eater.com": "Eater",
    "yelp.com": "Yelp",
    "timeout.com": "Time Out",
    "thrillist.com": "Thrillist",
    "bonappetit.com": "Bon Appétit",
    "nytimes.com": "NY Times",
    "sfchronicle.com": "SF Chronicle",
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

MAX_TEXT_LENGTH = 50000  # Increased to handle multi-page content
MAX_PAGES = 10  # Safety limit for pagination


def _get_source_name(url: str) -> str:
    hostname = urlparse(url).hostname or ""
    for domain, name in DOMAIN_NAMES.items():
        if hostname.endswith(domain):
            return name
    return hostname.removeprefix("www.")


def _fetch_page(url: str) -> requests.Response:
    resp = requests.get(url, headers=HEADERS, timeout=15)
    if resp.status_code in (429, 503):
        time.sleep(2)
        resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    return resp


def _clean_html(raw_html: str) -> str:
    soup = BeautifulSoup(raw_html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "iframe", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(line for line in lines if line)
    return text


def _extract_json_ld_restaurants(raw_html: str) -> list[str]:
    """Extract restaurant names from JSON-LD ItemList (used by Eater, etc.)."""
    soup = BeautifulSoup(raw_html, "html.parser")
    names = []
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string)
            if isinstance(data, dict) and data.get("@type") == "ItemList":
                for item in data.get("itemListElement", []):
                    if isinstance(item, dict):
                        name = item.get("name")
                        if name:
                            names.append(name)
        except (json.JSONDecodeError, TypeError):
            continue
    return names


def _find_pagination_urls(raw_html: str, base_url: str) -> list[str]:
    """Find pagination links in the page."""
    soup = BeautifulSoup(raw_html, "html.parser")
    page_urls = set()

    # Look for pagination containers
    for container in soup.find_all(["ul", "nav", "div"], class_=re.compile(r"paginat", re.I)):
        for link in container.find_all("a", href=True):
            href = urljoin(base_url, link["href"])
            if href != base_url and _is_same_site(base_url, href):
                page_urls.add(href)

    # Also look for next page links
    for link in soup.find_all("a", href=True):
        href = link.get("href", "")
        aria = link.get("aria-label", "").lower()
        text = link.get_text(strip=True).lower()
        if any(kw in aria or kw in text for kw in ["next", "page"]):
            full_url = urljoin(base_url, href)
            if full_url != base_url and _is_same_site(base_url, full_url):
                page_urls.add(full_url)

    # Check for Michelin-style /page/N pattern
    parsed = urlparse(base_url)
    path = parsed.path.rstrip("/")
    if "/page/" not in path:
        test_url = f"{parsed.scheme}://{parsed.netloc}{path}/page/2"
        page_urls.add(test_url)

    return sorted(page_urls)


def _is_same_site(url1: str, url2: str) -> bool:
    return urlparse(url1).netloc == urlparse(url2).netloc


def _fetch_with_pagination(url: str) -> tuple[str, int]:
    """Fetch a URL and follow pagination. Returns combined text and page count."""
    resp = _fetch_page(url)
    all_text_parts = [_clean_html(resp.text)]
    visited = {url.rstrip("/")}
    pages_fetched = 1

    # Check for JSON-LD (sites like Eater embed all data in first page)
    json_ld_names = _extract_json_ld_restaurants(resp.text)
    if json_ld_names:
        # Add JSON-LD names as a clear list for the LLM
        all_text_parts.append("\n--- Restaurant names from structured data ---\n")
        all_text_parts.append("\n".join(json_ld_names))

    # Find and follow pagination
    pagination_urls = _find_pagination_urls(resp.text, url)
    for page_url in pagination_urls:
        if page_url.rstrip("/") in visited:
            continue
        if pages_fetched >= MAX_PAGES:
            break
        try:
            time.sleep(1)  # Be polite between pages
            page_resp = _fetch_page(page_url)
            page_text = _clean_html(page_resp.text)
            all_text_parts.append(f"\n--- Page {pages_fetched + 1} ---\n")
            all_text_parts.append(page_text)
            visited.add(page_url.rstrip("/"))
            pages_fetched += 1

            # Check for more pagination on this page
            more_urls = _find_pagination_urls(page_resp.text, page_url)
            for more_url in more_urls:
                if more_url.rstrip("/") not in visited and more_url not in pagination_urls:
                    pagination_urls.append(more_url)
        except requests.RequestException:
            break  # Stop paginating on error

    combined = "\n".join(all_text_parts)
    return combined[:MAX_TEXT_LENGTH], pages_fetched


def fetch_url(url: str) -> FetchResult:
    source_name = _get_source_name(url)
    try:
        cleaned, pages = _fetch_with_pagination(url)
        return FetchResult(
            source_name=source_name,
            url=url,
            success=True,
            cleaned_text=cleaned,
            error_message="",
            pages_fetched=pages,
        )
    except requests.RequestException as e:
        return FetchResult(
            source_name=source_name,
            url=url,
            success=False,
            cleaned_text="",
            error_message=str(e),
            pages_fetched=0,
        )
