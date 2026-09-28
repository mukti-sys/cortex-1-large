"""
Test script to stream real industry datasets from Hugging Face:
1. princeton-nlp/SWE-bench_Lite (Real GitHub issues, stack traces, repo patches)
2. MickyMike/cvefixes_bigvul (Real CVE vulnerability diffs and clean fixes)
"""

import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from datasets import load_dataset

def test():
    print("=" * 60)
    print("STREAMING REAL INDUSTRY DATASETS FROM HUGGING FACE")
    print("=" * 60)

    # 1. SWE-bench Lite
    print("\n[1/2] Connecting to princeton-nlp/SWE-bench_Lite (streaming)...")
    ds_swe = load_dataset("princeton-nlp/SWE-bench_Lite", split="test", streaming=True)
    count_swe = 0
    for item in ds_swe:
        count_swe += 1
        print(f"  * Instance #{count_swe}: {item['repo']} - {item['instance_id']}")
        print(f"    Problem snippet: {item['problem_statement'][:120]}...")
        print(f"    Patch size: {len(item['patch'])} chars | Test patch: {len(item['test_patch'])} chars")
        if count_swe >= 3:
            break

    # 2. CVEFixes Big-Vul
    print("\n[2/2] Connecting to MickyMike/cvefixes_bigvul (streaming)...")
    ds_cve = load_dataset("MickyMike/cvefixes_bigvul", split="train", streaming=True)
    count_cve = 0
    for item in ds_cve:
        cwe = item.get("cwe_id")
        if cwe and cwe != "CWE-000":
            count_cve += 1
            print(f"  * CVE #{count_cve}: {cwe} (CVE: {item.get('cve_id')}) in {item.get('project_and_commit_id')[:30]}...")
            print(f"    Vulnerable source: {len(item.get('source', ''))} chars | Clean target: {len(item.get('target', ''))} chars")
            if count_cve >= 3:
                break

    print("\n[SUCCESS] Both real industry datasets verified and streamable!")
    print("=" * 60)

if __name__ == "__main__":
    test()
