"""
Generate LLM-powered matchup analysis using Claude.

Reads bracket picks from output/*_data.json, generates Claude-powered insights
for each matchup, and caches results. Falls back to existing template insights
when ANTHROPIC_API_KEY is not set.

Usage:
  export ANTHROPIC_API_KEY=sk-ant-...
  python scripts/generate_analysis.py                        # default bracket
  python scripts/generate_analysis.py --data output/index_data.json
  python scripts/generate_analysis.py --year 2025
"""

import json
import os
import sys
import time
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(ROOT, "data")

SYSTEM_PROMPT = """You are an elite college basketball analyst specializing in March Madness brackets. 
You provide concise, insightful analysis for individual tournament matchups.

Your analysis should:
- Be specific to the teams and their stats, not generic
- Reference concrete metrics (efficiency, tempo, defensive strength) when relevant
- Note historical patterns for the seed matchup
- Identify the key factor that will decide the game
- Be opinionated — clearly state who wins and why
- Sound like a knowledgeable friend giving bracket advice, not a textbook

Keep each analysis to 2-3 sentences. Be direct and punchy."""


def _build_matchup_prompt(pick, team_stats=None):
    """Build a Claude prompt for a single matchup."""
    parts = [
        f"Matchup: ({pick['seed_a']}) {pick['team_a']} vs ({pick['seed_b']}) {pick['team_b']}",
        f"Round: {pick['round_name']}",
        f"Region: {pick.get('region') or 'National'}",
        f"Model pick: {pick['pick']} (seed {pick['pick_seed']})",
        f"Win probability: {pick['win_prob']*100:.0f}%",
        (f"Spread: {pick['spread_fav']}, {pick['spread_dog']}" if pick.get('spread_fav') and pick.get('spread_dog') else f"Projected spread: {pick['projected_spread']} pts"),
        f"Projected score: {pick['projected_score']}",
        f"Confidence tier: {pick['confidence']}",
        f"Variability: {pick['variability']}",
    ]

    if pick.get("historical"):
        parts.append(f"Historical seed matchup record: {pick['historical']}")

    if team_stats:
        for team_key in ['team_a', 'team_b']:
            name = pick[team_key]
            stats = team_stats.get(name, {})
            if stats:
                seed_key = 'seed_a' if team_key == 'team_a' else 'seed_b'
                parts.append(
                    f"({pick[seed_key]}) {name}: "
                    f"AdjO={stats.get('adj_o', '?')}, AdjD={stats.get('adj_d', '?')}, "
                    f"Tempo={stats.get('adj_tempo', '?')}, BartHag={stats.get('barthag', '?')}"
                )

    if pick.get("key_factors"):
        parts.append("Key factors: " + "; ".join(pick["key_factors"]))

    parts.append(
        "\nProvide a 2-3 sentence analysis of this matchup. "
        "Also list 3-4 key factors as bullet points (short, ~8 words each). "
        "Format: first the analysis paragraph, then a blank line, then factors as '- factor text' lines."
    )
    return "\n".join(parts)


def generate_analysis_for_picks(picks, team_stats=None, cache_path=None, batch_size=5):
    """Generate Claude analysis for each pick. Returns updated picks list."""
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("  ANTHROPIC_API_KEY not set — using template analysis")
        return picks

    try:
        import anthropic
    except ImportError:
        print("  anthropic package not installed — using template analysis")
        return picks

    # Load cache
    cache = {}
    if cache_path and os.path.isfile(cache_path):
        with open(cache_path) as f:
            cache = json.load(f)
        print(f"  Loaded {len(cache)} cached analyses")

    client = anthropic.Anthropic(api_key=api_key)
    updated = 0
    total = len(picks)

    for i, pick in enumerate(picks):
        cache_key = f"{pick['team_a']}_vs_{pick['team_b']}_{pick['round']}"
        if cache_key in cache:
            _apply_cached(pick, cache[cache_key])
            continue

        prompt = _build_matchup_prompt(pick, team_stats)

        try:
            response = client.messages.create(
                model="claude-3-5-haiku-latest",
                max_tokens=300,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text.strip()
            parsed = _parse_response(text)
            _apply_cached(pick, parsed)
            cache[cache_key] = parsed
            updated += 1

            if (i + 1) % 10 == 0:
                print(f"  Generated {i+1}/{total} analyses...")

            # Rate limiting
            time.sleep(0.2)

        except Exception as e:
            print(f"  Error generating analysis for game {pick['game_num']}: {e}")
            continue

    # Save cache
    if cache_path and updated > 0:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(cache, f, indent=2)
        print(f"  Cached {updated} new analyses to {cache_path}")

    print(f"  {updated} new + {total - updated} cached analyses applied")
    return picks


def _parse_response(text):
    """Parse Claude's response into insight + factors."""
    lines = text.strip().split("\n")
    insight_lines = []
    factors = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("- ") or stripped.startswith("* "):
            factors.append(stripped[2:].strip())
        elif stripped:
            insight_lines.append(stripped)

    insight = " ".join(insight_lines).strip()
    return {"insight": insight, "key_factors": factors[:4]}


def _apply_cached(pick, cached):
    """Apply cached analysis to a pick."""
    if cached.get("insight"):
        pick["insight"] = cached["insight"]
    if cached.get("key_factors"):
        pick["key_factors"] = cached["key_factors"]


def main():
    parser = argparse.ArgumentParser(description="Generate Claude-powered matchup analysis")
    parser.add_argument("--data", type=str, default="output/index_data.json",
                        help="Path to bracket data JSON")
    parser.add_argument("--year", type=int, default=2026,
                        help="Year for cache file naming")
    parser.add_argument("--output", type=str, default=None,
                        help="Output updated data JSON (defaults to overwriting input)")
    args = parser.parse_args()

    data_path = os.path.join(ROOT, args.data) if not os.path.isabs(args.data) else args.data
    if not os.path.exists(data_path):
        print(f"ERROR: Data file not found: {data_path}")
        sys.exit(1)

    with open(data_path) as f:
        data = json.load(f)

    picks = data.get("bracket_picks", {}).get("picks", [])
    if not picks:
        print("No picks found in data file")
        sys.exit(1)

    print(f"Generating analysis for {len(picks)} matchups...")

    cache_path = os.path.join(DATA_DIR, f"analysis_cache_{args.year}.json")

    # Load team stats for richer prompts
    team_stats = {}
    for fname in [f"teams_merged_{args.year}.json", f"torvik_{args.year}.json"]:
        path = os.path.join(DATA_DIR, fname)
        if os.path.isfile(path):
            with open(path) as f:
                teams = json.load(f)
            if isinstance(teams, list):
                for t in teams:
                    if t.get("team"):
                        team_stats[t["team"]] = t
            break

    picks = generate_analysis_for_picks(picks, team_stats, cache_path)
    data["bracket_picks"]["picks"] = picks

    out_path = args.output or data_path
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
