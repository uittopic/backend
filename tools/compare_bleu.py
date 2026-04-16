#!/usr/bin/env python3
import json
import sys

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

v1_path = "outputs/batch_test/batch_test_results_20260415_125950.json"
v2_path = "outputs/batch_test/batch_test_results_20260416_040010.json"

v1_data = load_json(v1_path)
v2_data = load_json(v2_path)

print("=" * 80)
print("1. OVERALL STATS COMPARISON")
print("=" * 80)

v1_stats = v1_data['metadata']['stats']
v2_stats = v2_data['metadata']['stats']

print(f"\n{'Metric':<25} {'V1 (old)':<15} {'V2 (new)':<15} {'Change':<15}")
print("-" * 70)
print(f"{'Total Images':<25} {v1_stats['total_images']:<15} {v2_stats['total_images']:<15} {'0':<15}")
print(f"{'Success':<25} {v1_stats['success']:<15} {v2_stats['success']:<15} {'0':<15}")
print(f"{'Failed':<25} {v1_stats['failed']:<15} {v2_stats['failed']:<15} {'0':<15}")
print(f"{'Good':<25} {v1_stats['good_captions']:<15} {v2_stats['good_captions']:<15} {v2_stats['good_captions'] - v1_stats['good_captions']:<15}")
print(f"{'Medium':<25} {v1_stats['medium_captions']:<15} {v2_stats['medium_captions']:<15} {v2_stats['medium_captions'] - v1_stats['medium_captions']:<15}")
print(f"{'Bad':<25} {v1_stats['bad_captions']:<15} {v2_stats['bad_captions']:<15} {v2_stats['bad_captions'] - v1_stats['bad_captions']:<15}")
print(f"{'Device':<25} {v1_data['results'][0]['device']:<15} {v2_data['results'][0]['device']:<15}")

print("\n" + "=" * 80)
print("2. PROCESSING TIME ANALYSIS")
print("=" * 80)

def get_time_stats(results):
    times = [r['processing_time'] for r in results if r['success']]
    return min(times), sum(times)/len(times), max(times)

v1_min, v1_avg, v1_max = get_time_stats(v1_data['results'])
v2_min, v2_avg, v2_max = get_time_stats(v2_data['results'])

print(f"\n{'Metric':<25} {'V1 (mps)':<15} {'V2 (cpu)':<15}")
print("-" * 55)
print(f"{'Min (s)':<25} {v1_min:<15.3f} {v2_min:<15.3f}")
print(f"{'Avg (s)':<25} {v1_avg:<15.3f} {v2_avg:<15.3f}")
print(f"{'Max (s)':<25} {v1_max:<15.3f} {v2_max:<15.3f}")

print("\n" + "=" * 80)
print("3. QUALITY DEGRADATION ANALYSIS")
print("=" * 80)

# 定义质量分类函数
def classify_quality(caption, length):
    if length < 3:
        return 'bad'
    # 检查是否重复
    words = caption.split()
    if len(words) > 5:
        # 检查是否有重复词
        unique_words = set(words)
        if len(unique_words) / len(words) < 0.5:
            return 'medium'
    if length < 10:
        return 'medium'
    return 'good'

# 使用现有的统计数据，但我们需要分析每个样本的变化
# 创建V1和V2的映射
v1_map = {r['image_name']: r for r in v1_data['results']}
v2_map = {r['image_name']: r for r in v2_data['results']}

# 分析caption长度分布
print("\nCaption Length Distribution:")
print(f"{'Length Range':<20} {'V1 Count':<15} {'V2 Count':<15}")
print("-" * 50)

for range_start, range_end in [(0, 10), (10, 20), (20, 30), (30, 50), (50, 100), (100, 200)]:
    v1_count = sum(1 for r in v1_data['results'] if range_start <= len(r['caption_vi']) < range_end)
    v2_count = sum(1 for r in v2_data['results'] if range_start <= len(r['caption_vi']) < range_end)
    print(f"{range_start}-{range_end}:{'':<12} {v1_count:<15} {v2_count:<15}")

print("\n" + "=" * 80)
print("4. SAMPLE CAPTION COMPARISON (10 examples)")
print("=" * 80)

# 选择10个样本进行对比
sample_indices = [0, 10, 50, 100, 500, 1000, 2000, 3000, 4000, 5000]

for idx in sample_indices:
    v1_r = v1_data['results'][idx]
    v2_r = v2_data['results'][idx]
    img_name = v1_r['image_name']
    
    print(f"\n{'='*80}")
    print(f"Image: {img_name}")
    print(f"{'='*80}")
    print(f"V1: {v1_r['caption_vi']}")
    print(f"V2: {v2_r['caption_vi']}")
    
    v1_len = len(v1_r['caption_vi'])
    v2_len = len(v2_r['caption_vi'])
    print(f"Length - V1: {v1_len}, V2: {v2_len}")

print("\n" + "=" * 80)
print("5. CAPTION PATTERN ANALYSIS")
print("=" * 80)

def analyze_patterns(results, label):
    total_len = sum(len(r['caption_vi']) for r in results)
    avg_len = total_len / len(results)
    
    # 检查重复模式
    repetitive_count = 0
    for r in results:
        words = r['caption_vi'].split()
        if len(words) > 5:
            unique_words = set(words)
            if len(unique_words) / len(words) < 0.5:
                repetitive_count += 1
    
    # 检查常见关键词
    all_words = []
    for r in results:
        all_words.extend(r['caption_vi'].lower().split())
    
    return avg_len, repetitive_count

v1_avg_len, v1_repetitive = analyze_patterns(v1_data['results'], "V1")
v2_avg_len, v2_repetitive = analyze_patterns(v2_data['results'], "V2")

print(f"\n{'Metric':<25} {'V1':<15} {'V2':<15}")
print("-" * 55)
print(f"{'Avg Caption Length':<25} {v1_avg_len:<15.1f} {v2_avg_len:<15.1f}")
print(f"{'Repetitive Captions':<25} {v1_repetitive:<15} {v2_repetitive:<15}")
print(f"{'Repetitive %':<25} {v1_repetitive/len(v1_data['results'])*100:<14.2f}% {v2_repetitive/len(v2_data['results'])*100:<14.2f}%")

print("\n" + "=" * 80)
print("6. COMMON BAD CASES")
print("=" * 80)

print("\nV1 Bad Cases:")
v1_bad = [r for r in v1_data['results'] if 'bad' in str(r.get('quality_score', 'good')).lower()][:5]
for r in v1_bad[:3]:
    print(f"  - {r['image_name']}: {r['caption_vi']}")

print("\nV2 Bad Cases (V2 has 6 bad vs V1's 1):")
# V2的bad caption更多，需要找出
v2_samples_with_issues = []
for r in v2_data['results']:
    caption = r['caption_vi']
    words = caption.split()
    # 检查是否有问题
    if len(words) > 5:
        unique_words = set(words)
        if len(unique_words) / len(words) < 0.4:
            v2_samples_with_issues.append(r)

for r in v2_samples_with_issues[:5]:
    print(f"  - {r['image_name']}: {r['caption_vi']}")

print("\n" + "=" * 80)
print("7. KEY FINDINGS SUMMARY")
print("=" * 80)

print(f"""
📊 QUALITY CHANGES:
  - GOOD captions: {v1_stats['good_captions']:,} → {v2_stats['good_captions']:,} ({v2_stats['good_captions'] - v1_stats['good_captions']:,} decrease)
  - MEDIUM captions: {v1_stats['medium_captions']:,} → {v2_stats['medium_captions']:,} ({v2_stats['medium_captions'] - v1_stats['medium_captions']:,} increase)
  - BAD captions: {v1_stats['bad_captions']:,} → {v2_stats['bad_captions']:,} ({v2_stats['bad_captions'] - v1_stats['bad_captions']:,} increase)

⚡ PERFORMANCE:
  - V1 uses: mps (Apple Silicon GPU)
  - V2 uses: cpu
  - V2 is faster but quality is worse

📝 CAPTION PATTERNS:
  - V1: More detailed, specific product descriptions with more context
  - V2: More repetitive, generic, often includes words like "cà phê" (coffee) even for non-coffee items
  - V2 shows more hallucination/repetition issues
""")

print("=" * 80)