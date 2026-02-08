"""View trending topics from analysis results"""
import json
import sys

try:
    with open('output/python_trends.json', 'r') as f:
        trends = json.load(f)
    
    print("\n" + "=" * 80)
    print("  TOP TRENDING TOPICS - r/python Analysis")
    print("=" * 80)
    print(f"\n{'#':<4} {'Topic':<35} {'Total':<7} {'Recent':<8} {'Velocity':<10} {'Importance'}")
    print("-" * 80)
    
    for i, (topic, data) in enumerate(trends[:15], 1):
        print(f"{i:<4} {topic:<35} {data['total']:<7} {data['recent']:<8} {data['velocity']:<10.2f} {data['importance']:.2f}")
    
    print("=" * 80)
    print(f"\nTotal topics found: {len(trends)}")
    print("\nMetrics:")
    print("  - Total: Number of times topic appeared overall")
    print("  - Recent: Appearances in last 7 days")
    print("  - Velocity: Recent activity vs historical (higher = trending up)")
    print("  - Importance: Combined score (frequency + velocity)")
    print("=" * 80 + "\n")
    
except FileNotFoundError:
    print("Error: output/python_trends.json not found. Run an analysis first!")
    sys.exit(1)
