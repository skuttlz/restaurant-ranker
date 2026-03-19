import json
import os
from dataclasses import dataclass, field

from dotenv import load_dotenv
from openai import OpenAI

# Load from ~/.env for local dev; Railway injects env vars directly
load_dotenv(os.path.expanduser("~/.env"))


@dataclass
class RankedRestaurant:
    canonical_name: str
    mention_count: int
    sources: list[str] = field(default_factory=list)


def _get_client() -> OpenAI:
    return OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def extract_restaurants(source_name: str, city: str, cleaned_text: str) -> list[str]:
    client = _get_client()
    prompt = (
        f"Extract all restaurant names from the following text, which comes from "
        f"a restaurant review/recommendation page about {city}. "
        f"Return ONLY a JSON array of restaurant name strings. "
        f"Do not include descriptions, addresses, cuisine types, or any other information. "
        f"If you cannot find any restaurant names, return an empty array []."
    )
    resp = client.chat.completions.create(
        model="gpt-4o",
        temperature=0,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": cleaned_text},
        ],
    )
    content = resp.choices[0].message.content.strip()
    # Strip markdown code fences if present
    if content.startswith("```"):
        content = content.split("\n", 1)[1] if "\n" in content else content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
    try:
        names = json.loads(content)
        if isinstance(names, list):
            return [str(n) for n in names if n]
    except json.JSONDecodeError:
        pass
    return []


def deduplicate_and_rank(
    city: str, restaurants_by_source: dict[str, list[str]]
) -> list[RankedRestaurant]:
    if not restaurants_by_source:
        return []

    client = _get_client()
    prompt = (
        f"You are given restaurant names collected from multiple review sources for {city}. "
        f"Different sources may refer to the same restaurant with slightly different names "
        f"(e.g., 'Tartine Bakery' vs 'Tartine', 'State Bird Provisions' vs 'State Bird'). "
        f"Identify which names refer to the same restaurant and group them.\n\n"
        f"Return a JSON object where:\n"
        f"- Each key is the canonical (best/most complete) restaurant name\n"
        f"- Each value is an object with a 'sources' array listing which sources mentioned it\n\n"
        f"Input data (source -> restaurant names):\n"
        f"{json.dumps(restaurants_by_source, indent=2)}\n\n"
        f"Return ONLY the JSON object, no other text."
    )
    resp = client.chat.completions.create(
        model="gpt-4o",
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    content = resp.choices[0].message.content.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[1] if "\n" in content else content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

    try:
        grouped = json.loads(content)
    except json.JSONDecodeError:
        # Fallback: no dedup, just count raw mentions
        return _fallback_rank(restaurants_by_source)

    results = []
    for canonical_name, info in grouped.items():
        if isinstance(info, dict):
            sources = info.get("sources", [])
        elif isinstance(info, list):
            sources = info
        else:
            continue
        results.append(
            RankedRestaurant(
                canonical_name=canonical_name,
                mention_count=len(sources),
                sources=sorted(sources),
            )
        )

    results.sort(key=lambda r: (-r.mention_count, r.canonical_name))
    return results


def _fallback_rank(
    restaurants_by_source: dict[str, list[str]],
) -> list[RankedRestaurant]:
    name_sources: dict[str, set[str]] = {}
    for source, names in restaurants_by_source.items():
        for name in names:
            key = name.lower().strip()
            if key not in name_sources:
                name_sources[key] = set()
            name_sources[key].add(source)

    results = []
    for name, sources in name_sources.items():
        results.append(
            RankedRestaurant(
                canonical_name=name.title(),
                mention_count=len(sources),
                sources=sorted(sources),
            )
        )
    results.sort(key=lambda r: (-r.mention_count, r.canonical_name))
    return results
