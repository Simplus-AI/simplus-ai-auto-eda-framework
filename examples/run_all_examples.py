"""
Run All Examples Script
========================

This script runs all example scripts in sequence to demonstrate
the complete capabilities of the Simplus EDA Framework.

Run this to see everything in action:
    python examples/run_all_examples.py

Or run individual examples:
    python examples/00_complete_workflow.py
    python examples/01_quick_start_examples.py
    python examples/02_advanced_analysis_examples.py
    python examples/03_real_world_use_cases.py
"""

import sys
import os
import subprocess
from pathlib import Path


def run_script(script_name: str, description: str):
    """Run a Python script and display results"""
    print("\n\n")
    print("="*100)
    print(f"RUNNING: {description}")
    print("="*100)
    print(f"Script: {script_name}\n")

    script_path = Path(__file__).parent / script_name

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=False,
            text=True,
            check=True
        )
        print(f"\n✓ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {description} failed with error code {e.returncode}")
        return False
    except Exception as e:
        print(f"\n✗ Error running {description}: {e}")
        return False


def main():
    """Run all examples in sequence"""
    print("\n" + "="*100)
    print("SIMPLUS EDA FRAMEWORK - COMPLETE EXAMPLES SUITE")
    print("="*100)
    print("\nThis script will run all example demonstrations:")
    print("  0. Complete end-to-end workflow")
    print("  1. Quick start examples")
    print("  2. Advanced analysis features")
    print("  3. Real-world use cases")
    print("\nEstimated time: 5-10 minutes")
    print("="*100)

    # Ask for confirmation
    response = input("\nDo you want to continue? (y/N): ").strip().lower()
    if response not in ['y', 'yes']:
        print("\nAborted. You can run individual examples manually:")
        print("  python examples/00_complete_workflow.py")
        print("  python examples/01_quick_start_examples.py")
        print("  python examples/02_advanced_analysis_examples.py")
        print("  python examples/03_real_world_use_cases.py")
        return

    # Create outputs directory
    os.makedirs('outputs', exist_ok=True)

    # Track results
    results = {}

    # Run all examples
    examples = [
        ("00_complete_workflow.py", "Complete End-to-End Workflow"),
        ("01_quick_start_examples.py", "Quick Start Examples"),
        ("02_advanced_analysis_examples.py", "Advanced Analysis Examples"),
        ("03_real_world_use_cases.py", "Real-World Use Cases"),
    ]

    for script, description in examples:
        success = run_script(script, description)
        results[description] = success

    # Summary
    print("\n\n")
    print("="*100)
    print("EXAMPLES SUITE COMPLETE - SUMMARY")
    print("="*100)

    print("\n📊 Results:")
    for description, success in results.items():
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"  {status:12s} - {description}")

    total = len(results)
    passed = sum(results.values())
    print(f"\n{passed}/{total} examples completed successfully")

    if passed == total:
        print("\n🎉 All examples ran successfully!")
    else:
        print(f"\n⚠️  {total - passed} example(s) had issues")

    print("\n📁 Generated Reports:")
    print("  All reports have been saved to the outputs/ directory")
    print("  Open the HTML files in your browser to view the results")

    print("\n📖 What's in the outputs/ directory:")
    outputs_dir = Path('outputs')
    if outputs_dir.exists():
        html_files = sorted(outputs_dir.glob('*.html'))
        json_files = sorted(outputs_dir.glob('*.json'))

        if html_files:
            print("\n  HTML Reports:")
            for f in html_files:
                print(f"    • {f.name}")

        if json_files:
            print("\n  JSON Exports:")
            for f in json_files:
                print(f"    • {f.name}")

    print("\n🚀 Next Steps:")
    print("  1. Open HTML reports in your browser")
    print("  2. Review the console output above for insights")
    print("  3. Modify the examples to experiment with your own data")
    print("  4. Check out the documentation at docs/USAGE_GUIDE.md")
    print("  5. Read the examples README at examples/README.md")

    print("\n" + "="*100)
    print("Thank you for exploring the Simplus EDA Framework!")
    print("="*100 + "\n")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
