"""
Promptfoo JSON → SARIF Converter
----------------------------------
Converts Promptfoo's JSON output into SARIF 2.1.0 format so results
can be uploaded to the GitHub Security tab via upload-sarif action.

Usage:
    python prompts/promptfoo_to_sarif.py promptfoo-results.json output.sarif

GitHub Security tab integration requires:
    permissions:
        security-events: write
"""

import json
import sys
import datetime


def convert(input_path: str, output_path: str) -> None:
    try:
        with open(input_path) as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Warning: Could not read {input_path}: {e}. Writing empty SARIF.")
        data = {}

    results = data.get("results", [])

    sarif_results = []
    rules = []
    seen_rules: set[str] = set()

    for result in results:
        if not isinstance(result, dict):
            print(f"Warning: Expected dict result, got {type(result)}. Skipping.")
            continue

        description = result.get("description", "Unknown test")
        prompt = result.get("prompt", {})
        prompt_text = prompt.get("raw", "") if isinstance(prompt, dict) else str(prompt)
        test_cases = result.get("testResults", result.get("results", []))

        for tc in test_cases:
            for assertion in tc.get("namedScores", {}).items():
                metric_name, score = assertion
                passed = score >= 1.0
                if passed:
                    continue  # Only report failures to GitHub Security tab

                rule_id = f"promptfoo/{metric_name.replace('/', '-')}"

                if rule_id not in seen_rules:
                    seen_rules.add(rule_id)
                    rules.append({
                        "id": rule_id,
                        "name": metric_name.replace("/", " ").replace("-", " ").title(),
                        "shortDescription": {"text": f"Promptfoo compliance: {metric_name}"},
                        "fullDescription": {
                            "text": (
                                f"AI compliance assertion '{metric_name}' failed. "
                                f"This means the agent's output did not meet the expected behaviour "
                                f"for the test: {description}"
                            )
                        },
                        "helpUri": "https://promptfoo.dev/docs/configuration/expected-outputs",
                        "properties": {"tags": ["ai-compliance", "promptfoo"]},
                        "defaultConfiguration": {"level": "error"},
                    })

                sarif_results.append({
                    "ruleId": rule_id,
                    "level": "error",
                    "message": {
                        "text": (
                            f"FAILED: {description}\n"
                            f"Assertion: {metric_name}\n"
                            f"Prompt: {prompt_text[:200]}..."
                        )
                    },
                    "locations": [
                        {
                            "physicalLocation": {
                                "artifactLocation": {
                                    "uri": "prompts/promptfoo_config.yaml",
                                    "uriBaseId": "%SRCROOT%",
                                },
                                "region": {"startLine": 1},
                            }
                        }
                    ],
                })

    sarif = {
        "$schema": "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json",
        "version": "2.1.0",
        "runs": [
            {
                "tool": {
                    "driver": {
                        "name": "Promptfoo",
                        "version": "latest",
                        "informationUri": "https://promptfoo.dev",
                        "rules": rules,
                    }
                },
                "results": sarif_results,
                "invocations": [
                    {
                        "executionSuccessful": True,
                        "endTimeUtc": datetime.datetime.utcnow().isoformat() + "Z",
                    }
                ],
            }
        ],
    }

    with open(output_path, "w") as f:
        json.dump(sarif, f, indent=2)

    print(
        f"SARIF written to {output_path}: "
        f"{len(sarif_results)} failure(s) → "
        f"{len(rules)} rule(s) defined"
    )


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input.json> <output.sarif>")
        sys.exit(1)
    convert(sys.argv[1], sys.argv[2])
