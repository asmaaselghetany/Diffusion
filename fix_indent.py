#!/usr/bin/env python3
"""Fix indentation in extract_phase1a_metrics.py"""

with open('extract_phase1a_metrics.py', 'r') as f:
    lines = f.readlines()

# Fix lines 224-232
lines[223] = '        row = {\n'
lines[224] = '            "run_id": run_name,\n'
lines[225] = '            "phase": phase_dir.name,\n'
lines[226] = '        }\n'
lines[227] = '        \n'
lines[228] = '        # Add config parameters\n'
lines[229] = '        config = cfg_data.get("config", {})\n'
lines[230] = '        for key, value in config.items():\n'
lines[231] = '            row[f"config.{key}"] = value\n'
lines[232] = '        \n'

with open('extract_phase1a_metrics.py', 'w') as f:
    f.writelines(lines)

print("Fixed indentation")

