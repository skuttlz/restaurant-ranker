import json
import os
import re
from datetime import date
from pathlib import Path

import streamlit as st

from matcher import deduplicate_and_rank, extract_restaurants
from scrapers import fetch_url

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)


def _cache_key(city: str, urls: list[str]) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", city.lower()).strip("_")
    url_hash = hash(tuple(sorted(urls))) % 10**8
    return f"{slug}_{date.today().isoformat()}_{url_hash}"


def _load_cache(key: str):
    path = CACHE_DIR / f"{key}.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def _save_cache(key: str, data: dict):
    path = CACHE_DIR / f"{key}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def main():
    st.set_page_config(page_title="Restaurant Ranker", page_icon="🍽️", layout="wide")
    st.title("Restaurant Ranker")
    st.markdown(
        "Paste URLs to restaurant recommendation pages from different review sites. "
        "The app will extract restaurant names and rank them by how many sources mention each one."
    )

    # Sidebar
    with st.sidebar:
        st.header("Filters")
        min_mentions = st.slider("Minimum mentions", 1, 10, 1)
        force_refresh = st.checkbox("Force refresh (ignore cache)")

    # Main inputs
    city = st.text_input("City", placeholder="e.g., San Francisco")
    urls_text = st.text_area(
        "Paste URLs (one per line)",
        height=150,
        placeholder=(
            "https://guide.michelin.com/us/en/california/san-francisco/restaurants\n"
            "https://sf.eater.com/maps/best-restaurants-san-francisco\n"
            "https://www.theinfatuation.com/san-francisco/guides/best-restaurants-san-francisco"
        ),
    )

    if st.button("Find Restaurants", type="primary"):
        if not city.strip():
            st.error("Please enter a city name.")
            return
        urls = [u.strip() for u in urls_text.strip().splitlines() if u.strip()]
        if len(urls) < 2:
            st.error("Please provide at least 2 URLs.")
            return

        cache_key = _cache_key(city, urls)
        cached = None if force_refresh else _load_cache(cache_key)

        if cached:
            st.info("Loaded from cache. Check 'Force refresh' to re-fetch.")
            results = cached["results"]
            fetch_info = cached["fetch_info"]
        else:
            results, fetch_info = _run_pipeline(city, urls)
            _save_cache(cache_key, {"results": results, "fetch_info": fetch_info})

        _display_results(results, fetch_info, min_mentions)


def _run_pipeline(city: str, urls: list[str]):
    fetch_info = []
    restaurants_by_source = {}

    progress = st.progress(0, text="Starting...")
    total_steps = len(urls) + 1  # +1 for dedup step

    for i, url in enumerate(urls):
        progress.progress(
            (i) / total_steps,
            text=f"Fetching {url[:60]}...",
        )
        result = fetch_url(url)
        info = {
            "source": result.source_name,
            "url": url,
            "success": result.success,
            "error": result.error_message,
            "restaurant_count": 0,
            "pages_fetched": result.pages_fetched,
        }

        if result.success:
            with st.spinner(f"Extracting restaurants from {result.source_name}..."):
                names = extract_restaurants(result.source_name, city, result.cleaned_text)
                info["restaurant_count"] = len(names)
                if names:
                    restaurants_by_source[result.source_name] = names
        else:
            st.warning(f"Failed to fetch {result.source_name}: {result.error_message}")

        fetch_info.append(info)

    progress.progress(
        len(urls) / total_steps,
        text="Deduplicating and ranking...",
    )

    if restaurants_by_source:
        with st.spinner("Cross-referencing restaurants across sources..."):
            ranked = deduplicate_and_rank(city, restaurants_by_source)
    else:
        ranked = []

    progress.progress(1.0, text="Done!")

    results = [
        {
            "canonical_name": r.canonical_name,
            "mention_count": r.mention_count,
            "sources": r.sources,
        }
        for r in ranked
    ]
    return results, fetch_info


def _display_results(results: list[dict], fetch_info: list[dict], min_mentions: int):
    st.divider()

    # Filter by min mentions
    filtered = [r for r in results if r["mention_count"] >= min_mentions]

    if not filtered:
        st.warning("No restaurants found matching your criteria.")
        return

    st.subheader(f"🏆 {len(filtered)} Restaurants Ranked")

    # Build display table
    table_data = []
    for i, r in enumerate(filtered, 1):
        table_data.append(
            {
                "Rank": i,
                "Restaurant": r["canonical_name"],
                "Mentions": r["mention_count"],
                "Sources": ", ".join(r["sources"]),
            }
        )

    st.dataframe(
        table_data,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Rank": st.column_config.NumberColumn(width="small"),
            "Restaurant": st.column_config.TextColumn(width="large"),
            "Mentions": st.column_config.NumberColumn(width="small"),
            "Sources": st.column_config.TextColumn(width="medium"),
        },
    )

    # Source details
    st.subheader("Source Details")
    for info in fetch_info:
        status = "✅" if info["success"] else "❌"
        with st.expander(f"{status} {info['source']} — {info['restaurant_count']} restaurants"):
            st.write(f"**URL:** {info['url']}")
            if info["success"]:
                pages = info.get("pages_fetched", 1)
                pages_str = f" (across {pages} pages)" if pages > 1 else ""
                st.write(f"**Restaurants found:** {info['restaurant_count']}{pages_str}")
            else:
                st.error(f"Error: {info['error']}")


if __name__ == "__main__":
    main()
