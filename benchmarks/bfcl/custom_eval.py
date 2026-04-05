#!/usr/bin/env python3
"""
Custom BFCL Evaluator for NanoAgent

This script evaluates NanoAgent's BFCL responses by comparing with ground truth.
It parses the model's JSON tool calls and checks if they match the expected answers.
"""

import ast
import json
import re
from pathlib import Path
from typing import Any, Dict, List

from utils.tools import parse_tool_calls

CURR_FPATH = Path(__file__).parent.resolve()
BFCL_DATA_DIR = Path("/opt/homebrew/lib/python3.11/site-packages/bfcl_eval/data")

# MODEL_PATH = "weights/SmolLM2-135M-Instruct-nemotron-instruct-fc-instruct-sft"
# MODEL_PATH = "quwsarohi/NanoAgent-135M"
MODEL_PATH = "weights/NanoAgent-135M-nemotron-grpo-toolcall/best_checkpoint"
# MODEL_PATH = "weights/NanoAgent-135M-nemotron-sft"


RESULT_DIR = OUTPUT_DIR = CURR_FPATH.parent / 'results' / 'bfcl' / MODEL_PATH.split('/')[-1].lower().replace('-', '__')



def load_ground_truth(category: str) -> List[Dict]:
    """Load ground truth answers for a category."""
    filepath = BFCL_DATA_DIR / "possible_answer" / f"BFCL_v4_{category}.json"
    if not filepath.exists():
        return []
    
    data = []
    with open(filepath, "r") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def load_results(category: str) -> List[Dict]:
    """Load generated results for a category."""
    filepath = RESULT_DIR / f"BFCL_v4_{category}_result.json"
    if not filepath.exists():
        return []
    
    with open(filepath, "r") as f:
        return json.load(f)


def normalize_tool_call(tool_call: Dict) -> Dict:
    """Normalize a tool call for comparison."""
    normalized = {}
    for key, value in tool_call.items():
        if key == "name":
            normalized["name"] = value
        elif key == "arguments":
            normalized["arguments"] = {}
            if isinstance(value, dict):
                for arg_key, arg_value in value.items():
                    # Convert to list if single value
                    if not isinstance(arg_value, list):
                        arg_value = [arg_value]
                    normalized["arguments"][arg_key] = arg_value
        elif key == "arguments" and isinstance(value, str):
            # Try to parse string arguments
            try:
                parsed = json.loads(value)
                normalized["arguments"] = {}
                for arg_key, arg_value in parsed.items():
                    if not isinstance(arg_value, list):
                        arg_value = [arg_value]
                    normalized["arguments"][arg_key] = arg_value
            except:
                normalized["arguments"] = {}
    return normalized


def _parse_gt_string(gt_str: str) -> tuple:
    """Parse ground truth string like "func_name(arg1='value1', arg2='value2')"."""
    match = re.match(r"(\w+)\((.*)\)", gt_str)
    if not match:
        return None, {}
    func_name = match.group(1)
    args_str = match.group(2)
    
    args = {}
    arg_pattern = re.findall(r"(\w+)=([^,]+)", args_str)
    for key, value in arg_pattern:
        value = value.strip().strip("'\"")
        args[key] = value
    return func_name, args


def check_match(predicted: Dict, ground_truth: List) -> bool:
    """
    Check if predicted tool call matches any of the ground truth answers.
    
    Handles two formats:
    - Single-turn: [{function_name: {arg_name: [possible_values]}}]
    - Multi-turn: [["func(arg=value)", "func2(arg=value)"], [...]]
    """
    pred_normalized = normalize_tool_call(predicted)
    pred_name = pred_normalized.get("name", "")
    pred_args = pred_normalized.get("arguments", {})
    
    for gt in ground_truth:
        if isinstance(gt, str):
            gt_name, gt_args = _parse_gt_string(gt)
            if gt_name != pred_name:
                continue
            
            all_args_match = True
            for arg_name, expected_value in gt_args.items():
                if arg_name not in pred_args:
                    all_args_match = False
                    break
                
                pred_value = pred_args[arg_name][0] if pred_args[arg_name] else None
                
                if str(pred_value) != str(expected_value):
                    all_args_match = False
                    break
            
            if all_args_match:
                return True
        else:
            for func_name, func_args in gt.items():
                if func_name != pred_name:
                    continue
                
                all_args_match = True
                for arg_name, expected_values in func_args.items():
                    if arg_name not in pred_args:
                        all_args_match = False
                        break
                    
                    pred_value = pred_args[arg_name][0] if pred_args[arg_name] else None
                    
                    if pred_value not in expected_values:
                        all_args_match = False
                        break
                
                if all_args_match:
                    return True
    
    return False


def evaluate_category(category: str) -> Dict[str, Any]:
    """Evaluate a single category."""
    ground_truth = load_ground_truth(category)
    results = load_results(category)
    
    if not ground_truth:
        return {"error": f"No ground truth for {category}"}
    
    if not results:
        return {"error": f"No results for {category}"}
    
    # Create lookup for ground truth
    gt_lookup = {gt["id"]: gt.get("ground_truth", []) for gt in ground_truth}
    
    correct = 0
    total = len(results)
    tool_calls_made = 0
    parse_errors = 0
    
    for result in results:
        entry_id = result.get("id")
        
        # Handle different result formats
        raw_result = result.get("result", [])
        if isinstance(raw_result, list) and len(raw_result) > 0:
            if isinstance(raw_result[0], list):
                response = raw_result[0][0] if raw_result[0] else ""
            else:
                response = raw_result[0]
        else:
            response = ""
        
        # Parse tool calls from response
        tool_calls = parse_tool_calls(response)
        
        if tool_calls:
            tool_calls_made += 1
            
            # Check if any tool call matches ground truth
            matched = False
            for tc in tool_calls:
                if check_match(tc, gt_lookup.get(entry_id, [])):
                    matched = True
                    break
            
            if matched:
                correct += 1
        else:
            parse_errors += 1
    
    accuracy = (correct / total * 100) if total > 0 else 0
    tool_call_rate = (tool_calls_made / total * 100) if total > 0 else 0
    
    return {
        "category": category,
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "tool_calls_made": tool_calls_made,
        "tool_call_rate": tool_call_rate,
        "parse_errors": parse_errors,
    }


def evaluate_all(categories: List[str]) -> Dict[str, Any]:
    """Evaluate all categories."""
    all_results = {}
    
    for category in categories:
        print(f"Evaluating {category}...")
        result = evaluate_category(category)
        all_results[category] = result
        print(f"  Accuracy: {result.get('accuracy', 0):.1f}%")
        print(f"  Tool call rate: {result.get('tool_call_rate', 0):.1f}%")
    
    return all_results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", type=str, default=None, help="Specific category to evaluate")
    parser.add_argument("--output", type=str, default="eval_results.json", help="Output file name")
    args = parser.parse_args()
    
    # Define categories
    categories = [
        "simple_python",
        "simple_java", 
        "simple_javascript",
        "parallel",
        "multiple",
        "parallel_multiple",
        "irrelevance",
        "live_simple",
        "live_multiple",
        "live_parallel",
        "live_parallel_multiple",
        "live_irrelevance",
        "live_relevance",
        # "multi_turn_base",
        # "multi_turn_miss_func",
        # "multi_turn_miss_param",
        # "multi_turn_long_context",
        # "memory",
        # "web_search",
    ]
    
    if args.category:
        categories = [args.category]
    
    results = evaluate_all(categories)
    
    # Calculate overall scores
    total_correct = sum(r.get("correct", 0) for r in results.values() if "accuracy" in r)
    total_tests = sum(r.get("total", 0) for r in results.values() if "accuracy" in r)
    overall_accuracy = (total_correct / total_tests * 100) if total_tests > 0 else 0
    
    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    print(f"Overall Accuracy: {overall_accuracy:.1f}%")
    print(f"Total Tests: {total_tests}")
    print(f"Total Correct: {total_correct}")
    print("=" * 50)
    
    # Save results
    output_data = {
        "overall_accuracy": overall_accuracy,
        "total_tests": total_tests,
        "total_correct": total_correct,
        "categories": results,
    }
    
    output_file = OUTPUT_DIR / args.output
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
