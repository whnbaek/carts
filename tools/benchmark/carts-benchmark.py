import numpy as np
import matplotlib.pyplot as plt
import subprocess
import re
import json
import argparse
import os
import statistics
import glob
import csv
import datetime
import platform
import matplotlib
import yaml
import seaborn as sns
import sys
import shutil
import pandas as pd
import math
import logging
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any, Optional, Tuple, Union

# --- Basic Logging Setup ---
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Matplotlib and Seaborn Configuration ---
matplotlib.use('Agg')
sns.set_theme(style="whitegrid")
CUSTOM_COLORS = ["#1f77b4", "#ff7f0e"]  # blue, orange

# --- Performance Event Configuration ---
EVENT_GROUPS = {
    "Cache1": [
        "L1-dcache-loads:u", "L1-dcache-load-misses:u",
        "L1-dcache-stores:u", "L1-icache-load-misses:u",
        "cache-references:u", "cache-misses:u",
    ],
    "Cache2": [
        "L2-dcache-loads:u", "L2-dcache-load-misses:u",
        "l2_rqsts.all_code_rd:u", "l2_rqsts.code_rd_hit:u",
        "l2_rqsts.code_rd_miss:u", "l2_rqsts.miss:u",
    ],
    "Cache3": [
       "L3-dcache-loads:u", "L3-dcache-load-misses:u",
       "mem_inst_retired.all_loads:u", "mem_inst_retired.all_stores:u",
    ],
    "Clocks/Frequency": ["cycles:u", "cpu-clock:u", "task-clock:u"],
    "Stalls": ["stalled-cycles-frontend:u", "stalled-cycles-backend:u",
               "cycle_activity.stalls_total:u", "cycle_activity.stalls_mem_any:u",
               "resource_stalls.sb:u", "resource_stalls.scoreboard:u"],
    "TLB": ["dTLB-load-misses:u", "iTLB-load-misses:u", "dTLB-store-misses:u"],
    "Instructions": ["instructions:u"],
}

BASE_EVENTS = [
    "L1-dcache-loads", "L1-dcache-load-misses", "L1-dcache-stores", "L1-icache-load-misses",
    "l2_rqsts.all_code_rd", "l2_rqsts.code_rd_hit", "l2_rqsts.code_rd_miss", "l2_rqsts.miss",
    "dTLB-load-misses", "dTLB-store-misses", "iTLB-load-misses",
    "cycle_activity.stalls_total", "cycle_activity.stalls_mem_any", "resource_stalls.sb", "resource_stalls.scoreboard",
    "cycles", "instructions", "cpu-clock", "task-clock",
    "mem_inst_retired.all_loads", "mem_inst_retired.all_stores",
    "cache-references", "cache-misses"
]
DEFAULT_PERF_EVENTS = [e + ":u" for e in BASE_EVENTS]


# --- Helper Function (Module Level) ---
def safe_fmt(val: Optional[Union[float, int]], fmt: str = ".2f", na_val: str = "N/A") -> str:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return na_val
    try:
        return format(val, fmt)
    except (TypeError, ValueError):
        return str(val)

# --- ARTS Introspection Counter Parsing Helper (Module Level) ---
def parse_and_aggregate_counters(counters_dir: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Parses counter_*.ct files from the given directory.
    Returns a tuple: (aggregated_results_list, raw_file_data_list)
    
    aggregated_results_list: List of {"name": str, "aggregated_count": int, "aggregated_total_time": float}
    raw_file_data_list: List of {
        "file_name": str,
        "raw_entries": List of {
            "name": str, "nodeId": int, "threadId": int, "count": int, "totalTime": float
        }
    }
    Assumes each relevant line in a .ct file is space-separated after a header:
    name nodeId threadId count totalTime
    """
    aggregated_results_map: Dict[str, Dict[str, Union[int, float]]] = defaultdict(lambda: {"count": 0, "total_time": 0.0})
    raw_file_data_list: List[Dict[str, Any]] = []
    
    if not counters_dir.exists() or not counters_dir.is_dir():
        logger.warning(f"Counters directory {counters_dir} not found. Cannot parse counters.")
        return [], []

    ct_files = list(counters_dir.glob("counter_*.ct"))
    if not ct_files:
        logger.warning(f"No counter_*.ct files found in {counters_dir}.")
        return [], []

    logger.info(f"Parsing {len(ct_files)} counter files from {counters_dir}...")
    for ct_file_path in ct_files:
        current_file_raw_entries: List[Dict[str, Any]] = []
        try:
            with open(ct_file_path, 'r') as f:
                is_header = True 
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    
                    if is_header:
                        if line.startswith("name nodeId threadId count totalTime"):
                            is_header = False
                            continue
                        else:
                            logger.warning(f"File {ct_file_path} does not start with expected header. Attempting parse from line 1.")
                            is_header = False

                    parts = line.split()
                    if len(parts) == 5:
                        try:
                            name_val = parts[0].strip()
                            node_id_val = int(parts[1].strip())
                            thread_id_val = int(parts[2].strip())
                            count_val = int(parts[3].strip())
                            total_time_val = float(parts[4].strip())
                            
                            # For aggregation
                            aggregated_results_map[name_val]["count"] += count_val # type: ignore
                            aggregated_results_map[name_val]["total_time"] += total_time_val # type: ignore
                            
                            # For raw data
                            current_file_raw_entries.append({
                                "name": name_val,
                                "nodeId": node_id_val,
                                "threadId": thread_id_val,
                                "count": count_val,
                                "totalTime": total_time_val
                            })
                        except ValueError:
                            logger.warning(f"Skipping malformed data line {line_num} in {ct_file_path}: '{line}' (numeric conversion error)")
                        except Exception as e:
                            logger.warning(f"Error processing data line {line_num} in {ct_file_path}: '{line}' - {e}")
                    else:
                        logger.warning(f"Skipping line {line_num} in {ct_file_path} (unexpected columns {len(parts)}/5): '{line}'")
            
            if current_file_raw_entries:
                raw_file_data_list.append({
                    "file_name": ct_file_path.name,
                    "raw_entries": current_file_raw_entries
                })

        except Exception as e:
            logger.error(f"Could not read or process counter file {ct_file_path}: {e}")

    aggregated_output_list = [
        {"name": name, "aggregated_count": data["count"], "aggregated_total_time": data["total_time"]}
        for name, data in aggregated_results_map.items()
    ]
    logger.info(f"Aggregated {len(aggregated_output_list)} unique counters. Collected raw data from {len(raw_file_data_list)} files.")
    return aggregated_output_list, raw_file_data_list


# --- Manager Classes ---

class SystemInfoManager:
    """Manages gathering of system, reproducibility, and dependency information."""
    def __init__(self):
        self._system_info: Optional[Dict[str, Any]] = None
        self._reproducibility_info: Optional[Dict[str, Any]] = None
        self._dependency_versions: Optional[Dict[str, str]] = None

    def get_system_info(self) -> Dict[str, Any]:
        if self._system_info is None:
            info = {"timestamp": datetime.datetime.now().isoformat()}
            try:
                cpu_info_raw = subprocess.check_output("lscpu", text=True, stderr=subprocess.DEVNULL).strip()
                cpu_model = re.search(r"Model name:\s*(.*)", cpu_info_raw)
                cpu_cores = re.search(r"^CPU\(s\):\s*(\d+)", cpu_info_raw, re.MULTILINE)
                info["cpu_model"] = cpu_model.group(1).strip() if cpu_model else "N/A"
                info["cpu_cores"] = int(cpu_cores.group(1)) if cpu_cores else "N/A"
            except Exception as e:
                logger.warning(f"Could not get CPU info: {e}")
                info["cpu_model"], info["cpu_cores"] = "N/A", "N/A"
            try:
                mem_info_raw = subprocess.check_output("free -m", text=True, shell=True, stderr=subprocess.DEVNULL).strip()
                mem_total = re.search(r"Mem:\s*(\d+)", mem_info_raw)
                info["memory_total_mb"] = int(mem_total.group(1)) if mem_total else "N/A"
            except Exception as e:
                logger.warning(f"Could not get Memory info: {e}")
                info["memory_total_mb"] = "N/A"
            try:
                clang_version_raw = subprocess.check_output("clang --version", text=True, shell=True, stderr=subprocess.DEVNULL).strip()
                info["clang_version"] = clang_version_raw.splitlines()[0] if clang_version_raw else "N/A"
            except Exception as e:
                logger.warning(f"Could not get clang version: {e}")
                info["clang_version"] = "N/A"
            info["os_platform"] = platform.platform()
            info["os_uname"] = " ".join(platform.uname())
            self._system_info = info
        return self._system_info

    def _get_git_commit(self, path: Path) -> str:
        try:
            return subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return 'N/A'

    def get_dependency_versions(self, project_root: Path) -> Dict[str, str]:
        if self._dependency_versions is None:
            deps_paths = {
                'arts': project_root / 'external/arts',
                'polygeist': project_root / 'external/Polygeist',
                'polygeist_llvm': project_root / 'external/Polygeist/llvm-project/llvm',
            }
            versions = {}
            for name, path in deps_paths.items():
                versions[name] = self._get_git_commit(path)
            self._dependency_versions = versions
        return self._dependency_versions

    def get_reproducibility_info(self, project_root: Path) -> Dict[str, Any]:
        if self._reproducibility_info is None:
            info = {'python_version': sys.version.replace('\n', ' ')}
            try:
                pip_freeze = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True, stderr=subprocess.DEVNULL)
                info['pip_freeze'] = pip_freeze.strip().split('\n')
            except Exception as e:
                info['pip_freeze'] = [f"Error getting pip freeze: {e}"]
            info['env_vars'] = {k: v for k, v in os.environ.items() if k.startswith(('OMP_', 'CARTS_', 'PATH', 'LD_'))}
            info['git_commit_main_project'] = self._get_git_commit(project_root)
            self._reproducibility_info = info
        return self._reproducibility_info

class DirectoryManager:
    """Manages workspace directories and finding example configurations."""
    def __init__(self, project_root: Path, example_base_dirs_config: List[str], output_dir_name: str = "output"):
        self.project_root = project_root
        self.example_base_paths = [project_root / d for d in example_base_dirs_config]
        self.base_output_dir = project_root / output_dir_name
        self.output_dir: Optional[Path] = None

    def setup_output_directory(self, clear_existing: bool) -> Path:
        if clear_existing:
            self.output_dir = self.base_output_dir
            if self.output_dir.exists():
                self._clear_directory_contents(self.output_dir)
        else:
            self.output_dir = self._get_next_available_dir(self.base_output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output will be saved to directory: {self.output_dir.resolve()}")
        return self.output_dir

    def _clear_directory_contents(self, dir_path: Path):
        logger.info(f"Clearing contents of {dir_path}")
        for item in dir_path.iterdir():
            if item.name == '.gitignore':
                continue
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()

    def _get_next_available_dir(self, base_dir: Path) -> Path:
        if not base_dir.exists() or not any(base_dir.iterdir()):
            return base_dir
        i = 1
        while True:
            candidate_dir = Path(f"{base_dir}_{i}")
            if not candidate_dir.exists():
                return candidate_dir
            i += 1

    def find_example_directories(self, target_example_names: Optional[List[str]] = None) -> List[Path]:
        example_dirs = []
        for base_path in self.example_base_paths:
            if not base_path.is_dir():
                logger.warning(f"Example base path {base_path} does not exist or is not a directory.")
                continue
            for item_path in base_path.iterdir():
                if item_path.is_dir():
                    is_in_benchmarks_subdir = False
                    current_parent = item_path.parent
                    while current_parent != self.project_root and current_parent != current_parent.parent :
                        if current_parent.name == 'benchmarks':
                            if base_path.name != 'benchmarks':
                                is_in_benchmarks_subdir = True
                                break
                        current_parent = current_parent.parent
                    if is_in_benchmarks_subdir:
                        continue

                    config_file = item_path / 'test_config.yaml'
                    if config_file.is_file():
                        example_dirs.append(item_path)

        logger.info(f"Found {len(example_dirs)} example directories with test_config.yaml from bases {self.example_base_paths}.")

        if target_example_names:
            filtered_dirs = [d for d in example_dirs if d.name in target_example_names]
            if len(filtered_dirs) != len(target_example_names):
                found_names = {d.name for d in filtered_dirs}
                missing_targets = [t for t in target_example_names if t not in found_names]
                logger.warning(f"Could not find all specified target examples. Missing: {missing_targets}")
            example_dirs = filtered_dirs
            logger.info(f"Filtered to {len(example_dirs)} target examples: {[d.name for d in example_dirs]}")

        return example_dirs

class ConfigManager:
    """Manages loading and interpreting test configurations for a specific example."""
    def __init__(self, example_dir: Path, cli_args: Optional[argparse.Namespace] = None):
        self.example_dir = example_dir
        self.cli_args = cli_args
        self.test_config_path = example_dir / 'test_config.yaml'
        self.arts_cfg_path = example_dir / 'arts.cfg'
        self._test_config: Optional[Dict[str, Any]] = None
        self._arts_config_cache: Optional[Dict[str, Optional[Union[int, str]]]] = None

    def load_test_config(self) -> Dict[str, Any]:
        if self._test_config is None:
            if not self.test_config_path.is_file():
                logger.error(f"test_config.yaml not found in {self.example_dir}")
                return {}
            try:
                with open(self.test_config_path, 'r') as f:
                    config = yaml.safe_load(f)
                self._test_config = {
                    'benchmark': config.get('benchmark', {}),
                    'profiling': config.get('profiling', {})
                }
            except yaml.YAMLError as e:
                logger.error(f"Error parsing {self.test_config_path}: {e}")
                return {}
            except Exception as e:
                logger.error(f"Unexpected error loading {self.test_config_path}: {e}")
                return {}
        return self._test_config

    def get_benchmark_config(self) -> Dict[str, Any]:
        return self.load_test_config().get('benchmark', {})

    def get_profiling_config(self) -> Dict[str, Any]:
        return self.load_test_config().get('profiling', {})

    def is_profiling_enabled(self) -> bool:
        return self.get_profiling_config().get('profiling_enabled', False)

    def is_counter_enabled(self) -> bool:
        return self.get_profiling_config().get('counter_enabled', False)

    def get_run_plan(self) -> List[Tuple[str, int, str]]:
        bench_cfg = self.get_benchmark_config()
        problem_sizes = bench_cfg.get("problem_sizes", [])
        iterations_list = bench_cfg.get("iterations", [])
        args_list = bench_cfg.get("args", [""] * len(problem_sizes))

        if not isinstance(problem_sizes, list) or not isinstance(iterations_list, list) or not isinstance(args_list, list):
            logger.error(f"problem_sizes, iterations, or args is not a list in {self.test_config_path}")
            return []
        if not (len(problem_sizes) == len(iterations_list) == len(args_list)):
            logger.error(f"Mismatched lengths for problem_sizes, iterations, or args in {self.test_config_path}")
            return []
        return list(zip(map(str, problem_sizes), iterations_list, args_list))

    def get_threads_list(self) -> List[int]:
        return self.get_benchmark_config().get("threads", [1])

    def parse_arts_cfg(self) -> Dict[str, Optional[Union[int, str]]]:
        config: Dict[str, Optional[Union[int, str]]] = {"threads": None, "nodeCount": None}
        if not self.arts_cfg_path.exists():
            return config
        try:
            with open(self.arts_cfg_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("#") or "=" not in line:
                        continue
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip()
                    if key == "threads":
                        try: config["threads"] = int(value)
                        except ValueError: logger.warning(f"Non-integer 'threads' in {self.arts_cfg_path}: {value}")
                    elif key == "nodeCount":
                        try: config["nodeCount"] = int(value)
                        except ValueError: logger.warning(f"Non-integer 'nodeCount' in {self.arts_cfg_path}: {value}")
        except Exception as e:
            logger.error(f"Error parsing {self.arts_cfg_path}: {e}")
        return config

    def update_arts_cfg_threads(self, num_threads: int) -> None:
        lines = []
        found_threads = False
        if self.arts_cfg_path.exists():
            with open(self.arts_cfg_path, 'r') as f:
                for line in f:
                    if line.strip().startswith('threads =') or line.strip().startswith('threads='):
                        lines.append(f"threads = {num_threads}\n")
                        found_threads = True
                    else:
                        lines.append(line)
        if not found_threads:
            lines.append(f"threads = {num_threads}\n")
        try:
            with open(self.arts_cfg_path, 'w') as f:
                f.writelines(lines)
            logger.debug(f"Updated 'threads = {num_threads}' in {self.arts_cfg_path}")
        except Exception as e:
            logger.error(f"Error writing to {self.arts_cfg_path}: {e}")
        self._arts_config_cache = None

    def configure_arts_for_introspection(self, enable: bool):
        """Modifies arts.cfg to enable/disable ARTS introspection counter settings."""
        if not self.arts_cfg_path.exists() and enable:
            logger.warning(f"arts.cfg {self.arts_cfg_path} not found. Creating for introspection.")
            with open(self.arts_cfg_path, 'w') as f:
                f.write("counterFolder = ./counters\n")
                f.write("counterStartPoint = 1\n")
            logger.debug(f"Created and enabled introspection in {self.arts_cfg_path}")
            return
        elif not self.arts_cfg_path.exists():
            logger.debug(f"arts.cfg {self.arts_cfg_path} not found. No action for disabling introspection.")
            return

        lines = []
        with open(self.arts_cfg_path, 'r') as f:
            lines = f.readlines()

        original_lines_count = len(lines)
        lines = [line for line in lines if not line.strip().startswith(("counterFolder", "counterStartPoint"))]

        if enable:
            lines.append("counterFolder = ./counters\n")
            lines.append("counterStartPoint = 1\n")
            logger.debug(f"Enabled introspection settings in {self.arts_cfg_path}")
        else:
            if len(lines) < original_lines_count:
                logger.debug(f"Disabled introspection settings in {self.arts_cfg_path}")
            else:
                logger.debug(f"Introspection settings already disabled/absent in {self.arts_cfg_path}")

        with open(self.arts_cfg_path, 'w') as f:
            f.writelines(lines)
        self._arts_config_cache = None # Invalidate cache

    def get_grouped_perf_events(self) -> Optional[List[List[str]]]:
        profiling_section = self.get_profiling_config()
        extra_events = set()
        if self.cli_args and hasattr(self.cli_args, 'perf_events') and self.cli_args.perf_events:
            extra_events.update(self.cli_args.perf_events)
        
        yaml_perf_events = profiling_section.get('perf_events', [])
        if isinstance(yaml_perf_events, list):
            extra_events.update(yaml_perf_events)
        elif yaml_perf_events:
            logger.warning(f"perf_events in {self.test_config_path} should be a list. Ignoring.")

        def ensure_user_mode(ev: str) -> str:
            return ev if ev.endswith((":u", ":k")) else ev + ":u"

        all_defined_events = DEFAULT_PERF_EVENTS + [ensure_user_mode(e) for e in extra_events]
        unique_events = list(dict.fromkeys(all_defined_events))

        grouped_runs: List[List[str]] = []
        used_events = set()
        for _, group_config_events in EVENT_GROUPS.items():
            current_group_run = [ev for ev in group_config_events if ev in unique_events]
            if current_group_run:
                grouped_runs.append(current_group_run)
                used_events.update(current_group_run)
        remaining_events = [ev for ev in unique_events if ev not in used_events]
        if remaining_events:
            grouped_runs.append(remaining_events)
        return grouped_runs if any(grouped_runs) else None


class ExecutionManager:
    """Handles execution of make commands and program instances."""
    def __init__(self, timeout_seconds: int):
        self.timeout_seconds = timeout_seconds

    def run_make_command(self, make_target: str, cwd: Path) -> bool:
        logger.info(f"Running 'make {make_target}' in {cwd}...")
        try:
            process = subprocess.run(
                ["make", make_target], cwd=cwd, check=True, text=True, capture_output=True,
                timeout=self.timeout_seconds * 2
            )
            logger.debug(f"'make {make_target}' stdout: {process.stdout}")
            logger.info(f"'make {make_target}' successful in {cwd}.")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Error during 'make {make_target}' in {cwd}: {e.stderr}")
            return False
        except FileNotFoundError:
            logger.error(f"Error: 'make' command not found.")
            return False
        except subprocess.TimeoutExpired:
            logger.error(f"TIMEOUT during 'make {make_target}' in {cwd} after {self.timeout_seconds * 2}s.")
            return False

    def _run_single_program(self, command_list: List[str], cwd: Path, iteration_num: int, run_description: str = "") -> Tuple[Optional[float], str, str]:
        time_val: Optional[float] = None
        correctness = "UNKNOWN"
        full_output = ""
        cmd_str = ' '.join(command_list)
        logger.debug(f"Executing (Iter {iteration_num + 1}{' - ' + run_description if run_description else ''}): {cmd_str} in {cwd}")
        try:
            process = subprocess.run(
                command_list, capture_output=True, text=True, check=True, cwd=cwd,
                timeout=self.timeout_seconds
            )
            full_output = process.stdout + "\n" + process.stderr
            time_match = re.search(r"Finished in\s+([0-9]*\.?[0-9]+)\s*seconds?", full_output, re.IGNORECASE)
            if time_match:
                time_val = float(time_match.group(1))
            else:
                time_match_perf = re.search(r"([0-9]*\.?[0-9]+)\s+seconds time elapsed", full_output, re.IGNORECASE)
                if time_match_perf: time_val = float(time_match_perf.group(1))
                else: logger.warning(f"Iter {iteration_num + 1}{' - ' + run_description if run_description else ''}: Time not parsed. Output: {full_output[:200].replace(chr(10), ' ')}...")
            
            correctness_match = re.search(r"Result: (CORRECT|INCORRECT)", full_output, re.IGNORECASE)
            if correctness_match: correctness = correctness_match.group(1).upper()
            elif re.search(r"Result: OK", full_output, re.IGNORECASE): correctness = "CORRECT"
        except subprocess.TimeoutExpired:
            logger.warning(f"Iter {iteration_num + 1}{' - ' + run_description if run_description else ''}: TIMEOUT for {command_list[0]}.")
            correctness = "TIMEOUT"; full_output += f"\nTIMEOUT after {self.timeout_seconds}s."
        except subprocess.CalledProcessError as e:
            logger.error(f"Error running {command_list[0]} (iter {iteration_num + 1}{' - ' + run_description if run_description else ''}): {e.stderr or e.stdout}")
            full_output = (e.stdout or "") + "\n" + (e.stderr or ""); correctness = "ERROR_RUNTIME"
        except FileNotFoundError:
            logger.error(f"Executable {command_list[0]} not found."); correctness = "ERROR_MISSING_EXE"; full_output = f"Executable {command_list[0]} not found."
        except Exception as e:
            logger.error(f"Unexpected error running {command_list[0]} (iter {iteration_num + 1}{' - ' + run_description if run_description else ''}): {e}")
            correctness = "ERROR_UNEXPECTED"; full_output += f"\nUnexpected error: {e}"
        return time_val, correctness, full_output

    def run_benchmark_instance(self, executable_path: Path, program_args_str: str, num_iterations: int,
                               example_name: str, version_name: str) -> Dict[str, Any]:
        if not executable_path.exists():
            logger.error(f"Executable {executable_path} not found for {version_name} {example_name}. Skipping benchmark.")
            return {"times_seconds": [], "iterations_ran": 0, "iterations_requested": num_iterations, "correctness_status": "NOT_RUN (Executable Missing)", "all_outputs_concatenated": ""}
        logger.info(f"Running benchmark: {version_name} ('{executable_path.name}') for '{example_name}', args='{program_args_str}', iters={num_iterations}...")
        times: List[Optional[float]] = []; final_correctness = "UNKNOWN"; all_outputs : List[str] = []
        command = [str(executable_path)] + program_args_str.split()
        for i in range(num_iterations):
            time_val, iter_correctness, iter_output = self._run_single_program(command, executable_path.parent, i, f"{version_name} benchmark")
            times.append(time_val); all_outputs.append(iter_output)
            if iter_correctness != "UNKNOWN":
                 if final_correctness == "UNKNOWN" or final_correctness == "CORRECT" or iter_correctness != "CORRECT":
                    final_correctness = iter_correctness
            logger.debug(f"  Iter {i+1} ({version_name} benchmark): {safe_fmt(time_val, '.6f')}s, Status: {iter_correctness}")
        valid_times = [t for t in times if t is not None]
        return {"times_seconds": valid_times, "iterations_ran": len(valid_times), "iterations_requested": num_iterations,
                "correctness_status": final_correctness, "all_outputs_concatenated": "\n--- ITERATION END ---\n".join(all_outputs)}

    def run_profiling_instance(self, executable_path: Path, program_args_str: str, num_iterations: int,
                                example_name: str, version_name: str, perf_event_groups: List[List[str]]) -> Dict[str, Any]:
        if not executable_path.exists():
            logger.error(f"Executable {executable_path} not found for {version_name} {example_name}. Skipping profiling.")
            return {"event_stats": {}, "times_seconds": [], "iterations_ran": 0, "iterations_requested": num_iterations, "correctness_status": "NOT_RUN (Executable Missing)", "all_outputs_concatenated": ""}
        logger.info(f"Running profiling: {version_name} ('{executable_path.name}') for '{example_name}', args='{program_args_str}', iters={num_iterations}...")
        event_data_accumulator: Dict[str, List[float]] = defaultdict(list); times_from_perf_runs: List[Optional[float]] = []
        final_correctness = "UNKNOWN"; all_outputs_profiling: List[str] = []
        program_arg_list = program_args_str.split()
        for i in range(num_iterations):
            iter_correctness_overall = "UNKNOWN"; time_for_this_iter: Optional[float] = None; iter_output_combined_groups = ""
            for group_idx, event_group in enumerate(perf_event_groups):
                if not event_group: continue
                perf_cmd = ["perf", "stat", "-a", "-x", ",", "-e", ','.join(event_group), "--", str(executable_path)] + program_arg_list
                run_desc = f"{version_name} profiling group {group_idx+1}/{len(perf_event_groups)}"
                time_val, iter_correctness_group, perf_output_single_group = self._run_single_program(perf_cmd, executable_path.parent, i, run_desc)
                iter_output_combined_groups += f"--- Perf Group {group_idx+1} Output ---\n{perf_output_single_group}\n"
                if group_idx == 0:
                    time_for_this_iter = time_val
                    if iter_correctness_group != "UNKNOWN":
                        if iter_correctness_overall == "UNKNOWN" or iter_correctness_overall == "CORRECT" or iter_correctness_group != "CORRECT":
                             iter_correctness_overall = iter_correctness_group
                for line in perf_output_single_group.splitlines():
                    parts = line.strip().split(",")
                    if len(parts) >= 3 and not parts[0].strip().startswith(("#", "<not counted>", "<not supported>")):
                        try:
                            val_str = parts[0].strip()
                            val = float('nan') if val_str.lower() == 'nan' or val_str in ['<not supported>', '<not counted>'] else float(val_str)
                            event_data_accumulator[parts[2].strip()].append(val)
                        except (ValueError, IndexError): pass
            all_outputs_profiling.append(iter_output_combined_groups); times_from_perf_runs.append(time_for_this_iter)
            if iter_correctness_overall != "UNKNOWN":
                if final_correctness == "UNKNOWN" or final_correctness == "CORRECT" or iter_correctness_overall != "CORRECT":
                    final_correctness = iter_correctness_overall
            logger.debug(f"  Iter {i+1} ({version_name} profiling): {safe_fmt(time_for_this_iter, '.6f')}s, Status: {iter_correctness_overall}")
        processed_event_stats = {event: {"all_values": [v for v in values if not math.isnan(v)]} for event, values in event_data_accumulator.items()}
        valid_times = [t for t in times_from_perf_runs if t is not None]
        return {"event_stats": processed_event_stats, "times_seconds": valid_times, "iterations_ran": len(valid_times),
                "iterations_requested": num_iterations, "correctness_status": final_correctness, "all_outputs_concatenated": "\n--- ITERATION END ---\n".join(all_outputs_profiling)}

    def run_arts_inspector(self, inspector_path: Path, program_args_str: str, cwd: Path) -> bool:
        if not inspector_path.exists():
            logger.error(f"ARTS Inspector executable {inspector_path} not found. Skipping.")
            return False
        logger.info(f"Running ARTS Inspector: {inspector_path.name} with args '{program_args_str}' in {cwd}")
        command = [str(inspector_path)] + program_args_str.split()
        _time_val, status, output = self._run_single_program(command, cwd, 0, "ARTS Inspector")
        if status not in ["CORRECT", "OK", "UNKNOWN", "ERROR_RUNTIME"]:
            if status in ["TIMEOUT", "ERROR_MISSING_EXE", "ERROR_UNEXPECTED"]:
                 logger.error(f"ARTS Inspector critical fail: {status}. Output: {output[:500]}")
                 return False
            else:
                 logger.warning(f"ARTS Inspector completed with status: {status}. Output: {output[:500]}")
                 return True 
        # logger.info(f"ARTS Inspector run finished. Status: {status}. Output: {output[:200]}...")
        return True


class ResultsManager:
    """Manages collection, aggregation, and reporting of benchmark results."""
    def __init__(self):
        self._benchmark_runs: List[Dict[str, Any]] = []
        self._profiling_runs: List[Dict[str, Any]] = []
        self._arts_introspection_runs: List[Dict[str, Any]] = [] 

    def add_benchmark_run_data(self, data: Dict[str, Any]):
        self._benchmark_runs.append(data)

    def add_profiling_run_data(self, data: Dict[str, Any]):
        self._profiling_runs.append(data)

    def add_arts_introspection_run(self, example_name: str, problem_size: str, threads: int, 
                                   aggregated_counters: List[Dict[str, Any]], 
                                   raw_file_details: List[Dict[str, Any]]):
        self._arts_introspection_runs.append({
            "example_name": example_name,
            "problem_size": problem_size,
            "threads": threads,
            "aggregated_counters": aggregated_counters,
            "raw_file_details": raw_file_details
        })

    def get_raw_benchmark_runs(self) -> List[Dict[str, Any]]:
        return self._benchmark_runs

    def get_raw_profiling_runs(self) -> List[Dict[str, Any]]:
        return self._profiling_runs

    def get_arts_introspection_runs(self) -> List[Dict[str, Any]]: 
        return self._arts_introspection_runs

    def _compute_runtime_stats(self, times: List[float]) -> Dict[str, Optional[float]]:
        if not times: return {"min": None, "max": None, "avg": None, "median": None, "stdev": None}
        return {"min": min(times), "max": max(times), "avg": statistics.mean(times), 
                "median": statistics.median(times), "stdev": statistics.stdev(times) if len(times) > 1 else 0.0}

    def _compute_event_stats(self, values: List[float]) -> Dict[str, Optional[float]]:
        if not values: return {"min": None, "max": None, "avg": None, "median": None, "stdev": None}
        return self._compute_runtime_stats(values)

    def prepare_aggregated_data(self) -> List[Dict[str, Any]]:
        key_func = lambda r: (r["example_name"], r["problem_size"], r.get("threads", 1))
        processed_runs: Dict[Tuple, Dict[str, Any]] = defaultdict(lambda: {
            "CARTS_benchmark_data": None, "OMP_benchmark_data": None,
            "CARTS_profiling_data": None, "OMP_profiling_data": None, "carts_config_details": {}
        })
        all_event_names = set()
        for run in self._benchmark_runs:
            key = key_func(run); version = run["version"].upper()
            processed_runs[key][f"{version}_benchmark_data"] = run
            if version == "CARTS" and run.get("carts_config"): processed_runs[key]["carts_config_details"] = run.get("carts_config", {})
        for run in self._profiling_runs:
            key = key_func(run); version = run["version"].upper()
            processed_runs[key][f"{version}_profiling_data"] = run
            if version == "CARTS" and not processed_runs[key]["carts_config_details"] and run.get("carts_config"):
                 processed_runs[key]["carts_config_details"] = run.get("carts_config", {})
            if "event_stats" in run: all_event_names.update(run["event_stats"].keys())
        
        sorted_event_names = sorted(list(all_event_names)); output_rows = []
        for key_tuple, data_group in sorted(processed_runs.items()):
            ex_name, prob_size, threads = key_tuple; carts_cfg = data_group.get("carts_config_details", {})
            row: Dict[str, Any] = {"Example Name": ex_name, "Problem Size": prob_size, "Threads": threads,
                                   "CARTS Config Actual Threads": carts_cfg.get("threads", "N/A"),
                                   "CARTS Config Actual Nodes": carts_cfg.get("nodeCount", "N/A")}
            carts_b = data_group.get("CARTS_benchmark_data"); omp_b = data_group.get("OMP_benchmark_data")
            carts_t = carts_b.get("times_seconds", []) if carts_b else []; omp_t = omp_b.get("times_seconds", []) if omp_b else []
            carts_stats = self._compute_runtime_stats(carts_t); omp_stats = self._compute_runtime_stats(omp_t)
            
            row.update({"CARTS Median Time (s)": carts_stats["median"], "CARTS Avg Time (s)": carts_stats["avg"], "CARTS Min Time (s)": carts_stats["min"], "CARTS Max Time (s)": carts_stats["max"], "CARTS StdDev Time (s)": carts_stats["stdev"], "CARTS Times (s)": carts_t, "CARTS Correctness": carts_b.get("correctness_status", "N/A") if carts_b else "N/A", "CARTS Iterations Ran": carts_b.get("iterations_ran",0) if carts_b else 0, "CARTS Iterations Requested": carts_b.get("iterations_requested",0) if carts_b else 0})
            row.update({"OMP Median Time (s)": omp_stats["median"], "OMP Avg Time (s)": omp_stats["avg"], "OMP Min Time (s)": omp_stats["min"], "OMP Max Time (s)": omp_stats["max"], "OMP StdDev Time (s)": omp_stats["stdev"], "OMP Times (s)": omp_t, "OMP Correctness": omp_b.get("correctness_status", "N/A") if omp_b else "N/A", "OMP Iterations Ran": omp_b.get("iterations_ran",0) if omp_b else 0, "OMP Iterations Requested": omp_b.get("iterations_requested",0) if omp_b else 0})
            
            row["Speedup (OMP/CARTS)"] = (omp_stats["median"] / carts_stats["median"]) if carts_stats["median"] and omp_stats["median"] and carts_stats["median"] > 1e-9 else None
            
            for ver_prefix in ["CARTS", "OMP"]:
                prof_run = data_group.get(f"{ver_prefix}_profiling_data"); events_raw = prof_run.get("event_stats", {}) if prof_run else {}
                for event_n in sorted_event_names:
                    vals = events_raw.get(event_n, {}).get("all_values", [])
                    event_sum_stats = self._compute_event_stats(vals)
                    for stat_s in ["avg", "median", "min", "max", "stdev"]:
                        row[f"{ver_prefix} Event {event_n} {stat_s.capitalize()}"] = event_sum_stats[stat_s]
            output_rows.append(row)
        return output_rows

    def compute_overall_statistics(self, aggregated_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        speedups = [r["Speedup (OMP/CARTS)"] for r in aggregated_data if r.get("Speedup (OMP/CARTS)") is not None and not (isinstance(r["Speedup (OMP/CARTS)"], float) and math.isnan(r["Speedup (OMP/CARTS)"]))]
        carts_wins, omp_wins, ties = 0,0,0
        max_s_val, min_s_val, max_slow_val = None, None, None
        max_s_ex, min_s_ex, max_slow_ex = None, None, None

        for r in aggregated_data:
            s = r.get("Speedup (OMP/CARTS)")
            if s is not None and not (isinstance(s, float) and math.isnan(s)):
                cfg_str = f"{r['Example Name']} (Size: {r['Problem Size']}, Threads: {r['Threads']})"
                if s > 1.05: carts_wins +=1; 
                elif s < (1/1.05): omp_wins += 1
                else: ties += 1
                if max_s_val is None or s > max_s_val: max_s_val, max_s_ex = s, cfg_str
                if s < (1/1.05) and (max_slow_val is None or s < max_slow_val) : max_slow_val, max_slow_ex = s, cfg_str # s is small (CARTS slower)
                if s > 0 and (min_s_val is None or s < min_s_val): min_s_val, min_s_ex = s, cfg_str
        
        geom_mean = None
        valid_s_geom = [s for s in speedups if s > 0]
        if valid_s_geom:
            try: geom_mean = statistics.geometric_mean(valid_s_geom)
            except statistics.StatisticsError: logger.warning("Geom. mean error for speedups.")
        
        correctness = defaultdict(int)
        for run in self._benchmark_runs: correctness[f"{run['version'].upper()}_{run.get('correctness_status', 'UNKNOWN')}"] += 1
        
        return {"geometric_mean_speedup": geom_mean, "total_configurations_compared": len(speedups),
                "carts_faster_configs": carts_wins, "omp_faster_configs": omp_wins, "tied_configs": ties,
                "max_speedup_carts_faster": {"value": max_s_val, "config": max_s_ex},
                "max_speedup_omp_faster": {"value": 1.0/max_slow_val if max_slow_val and max_slow_val > 1e-9 else None, "config": max_slow_ex, "original_omp_carts_ratio": max_slow_val},
                "min_positive_speedup_ratio": {"value": min_s_val, "config": min_s_ex},
                "correctness_summary": dict(correctness), "all_speedups_raw": speedups}

    def collect_failure_report(self, aggregated_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        failures = []
        fail_stati = ["INCORRECT", "TIMEOUT", "ERROR_RUNTIME", "ERROR_MISSING_EXE", "ERROR_UNEXPECTED"]
        for r in aggregated_data:
            for ver_label, ver_key in [("CARTS", "CARTS"), ("OpenMP", "OMP")]:
                corr = r.get(f"{ver_key} Correctness")
                if corr and corr in fail_stati:
                    failures.append({"example": r["Example Name"], "problem_size": r["Problem Size"], "threads": r["Threads"],
                                     "version": ver_label, "status": corr, "iterations_ran": r.get(f"{ver_key} Iterations Ran", "N/A")})
        return failures

    def write_csv_report(self, aggregated_data: List[Dict[str, Any]], output_path: Path) -> None:
        if not aggregated_data: logger.info("No aggregated data for CSV."); return
        fnames = list(aggregated_data[0].keys())
        common = ["Example Name", "Problem Size", "Threads", "CARTS Config Actual Threads", "CARTS Config Actual Nodes", "Speedup (OMP/CARTS)", "CARTS Median Time (s)", "OMP Median Time (s)", "CARTS Correctness", "OMP Correctness"]
        ordered_fnames = [f for f in common if f in fnames] + sorted([f for f in fnames if f not in common])
        with open(output_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=ordered_fnames, restval="N/A", extrasaction='ignore')
            writer.writeheader()
            for row in aggregated_data:
                fmt_row = {}
                for k, v in row.items():
                    if isinstance(v, float): fmt_row[k] = safe_fmt(v, ".6f" if "Time (s)" in k or "Speedup" in k else ".2f")
                    elif k.endswith("Times (s)") and isinstance(v, list): fmt_row[k] = "; ".join([safe_fmt(t, ".6f") for t in v])
                    elif isinstance(v, list): fmt_row[k] = "; ".join([str(x) for x in v])
                    else: fmt_row[k] = v if v is not None else "N/A"
                writer.writerow(fmt_row)
        logger.info(f"Aggregated CSV report saved to {output_path}")

    def write_json_report(self, final_json_data: Dict[str, Any], output_path: Path) -> None:
        def default_serializer(obj):
            if isinstance(obj, Path): return str(obj)
            if isinstance(obj, (np.integer, np.floating, np.bool_)): return obj.item()
            if isinstance(obj, float) and math.isnan(obj): return "NaN"
            if isinstance(obj, float) and math.isinf(obj): return "Infinity" if obj > 0 else "-Infinity"
            raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable: {obj}")
        with open(output_path, "w") as f:
            json.dump(final_json_data, f, indent=2, default=default_serializer)
        logger.info(f"JSON report saved to {output_path}")

    def save_figure(self, fig, img_path: Path, pdf_path: Path):
        try:
            fig.savefig(img_path, bbox_inches='tight', dpi=300); logger.info(f"Saved PNG to {img_path}")
            fig.savefig(pdf_path, bbox_inches='tight'); logger.info(f"Saved PDF to {pdf_path}")
        except Exception as e: logger.error(f"Error saving figure: {e}")
        plt.close(fig)


# --- Main Orchestration Logic ---
def main():
    logger.info("- START CARTS Performance and Introspection Script -")
    parser = argparse.ArgumentParser(description="Run CARTS examples, benchmarks, profiling, and introspection.")
    parser.add_argument("--timeout_seconds", type=int, default=300, help="Timeout for each program run.")
    parser.add_argument("--output_prefix", type=str, default="performance_results", help="Prefix for output files.")
    parser.add_argument("--example_base_dirs", nargs='+', default=["examples/parallel", "examples/tasking"], help="Base example dirs.")
    parser.add_argument("--target_examples", nargs='+', default=None, help="Specific examples to run.")
    parser.add_argument('--clear_output', action=argparse.BooleanOptionalAction, default=True, help='Clear/reuse output dir vs. new one.')
    parser.add_argument('--perf_events', nargs='+', default=None, help='CLI override for perf events.')
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    dir_manager = DirectoryManager(project_root, args.example_base_dirs, "output_benchmarks")
    output_dir = dir_manager.setup_output_directory(args.clear_output)

    sys_info_manager = SystemInfoManager()
    exec_manager = ExecutionManager(args.timeout_seconds)
    results_manager = ResultsManager() 

    example_paths_to_run = dir_manager.find_example_directories(args.target_examples)
    if not example_paths_to_run: logger.info("No examples found. Exiting."); return
    run_id_counter = 1

    for example_path in example_paths_to_run:
        example_name = example_path.name
        logger.info(f"\n>>> Processing example: {example_name} in {example_path} <<<")
        config_manager = ConfigManager(example_path, args)
        if not config_manager.load_test_config():
            logger.warning(f"Skipping {example_name}: invalid test_config.yaml."); continue

        run_plan = config_manager.get_run_plan()
        threads_list = config_manager.get_threads_list()
        is_profiling_on = config_manager.is_profiling_enabled()
        perf_groups = config_manager.get_grouped_perf_events() if is_profiling_on else None
        is_arts_introspection_on = config_manager.is_counter_enabled()

        if not run_plan: logger.warning(f"Skipping {example_name}: invalid run plan."); continue

        for common_threads in threads_list:
            logger.info(f"--- Config: {example_name}, Threads={common_threads} ---")
            config_manager.update_arts_cfg_threads(common_threads)
            os.environ["OMP_NUM_THREADS"] = str(common_threads)

            if not exec_manager.run_make_command("clean", example_path) or \
               not exec_manager.run_make_command("all", example_path):
                logger.error(f"Make failed for {example_name}, threads={common_threads}. Skipping this thread config."); continue
            
            current_arts_cfg = config_manager.parse_arts_cfg()

            for prob_size, num_iters, spec_args in run_plan:
                logger.info(f"--- Run: Problem='{prob_size}', Threads={common_threads} ---")
                exec_map = {"CARTS": example_path / example_name, "OMP": example_path / f"{example_name}_omp"}
                run_args = " ".join([arg for arg in prob_size.split() if arg] + [arg for arg in spec_args.split() if arg])

                for ver_name, exec_file in exec_map.items():
                    bench_data = exec_manager.run_benchmark_instance(exec_file, run_args, num_iters, example_name, ver_name)
                    results_manager.add_benchmark_run_data({
                        "run_id": run_id_counter, "example_name": example_name, "problem_size": prob_size, "threads": common_threads,
                        "version": ver_name, "run_type": "benchmark", "carts_config": current_arts_cfg if ver_name == "CARTS" else {},
                        "iterations_requested": num_iters, **bench_data})
                    run_id_counter += 1

                    if is_profiling_on and perf_groups:
                        prof_data = exec_manager.run_profiling_instance(exec_file, run_args, num_iters, example_name, ver_name, perf_groups)
                        results_manager.add_profiling_run_data({
                            "run_id": run_id_counter, "example_name": example_name, "problem_size": prob_size, "threads": common_threads,
                            "version": ver_name, "run_type": "profiling", "carts_config": current_arts_cfg if ver_name == "CARTS" else {},
                            "iterations_requested": num_iters, **prof_data})
                        run_id_counter += 1
                
                if is_arts_introspection_on:
                    logger.info(f"ARTS Introspection: {example_name}, Size: {prob_size}, Threads: {common_threads}")
                    inspector_exec = example_path / f"{example_name}_inspector"
                    if inspector_exec.is_file():
                        config_manager.configure_arts_for_introspection(enable=True)
                        counters_dir = example_path / "counters"
                        if counters_dir.exists(): shutil.rmtree(counters_dir)
                        counters_dir.mkdir(parents=True, exist_ok=True)
                        
                        inspector_ok = exec_manager.run_arts_inspector(inspector_exec, run_args, example_path)
                        if inspector_ok:
                            # parse_and_aggregate_counters now returns two lists
                            aggregated_counters_list, raw_details_list = parse_and_aggregate_counters(counters_dir)
                            if aggregated_counters_list or raw_details_list: # Add if either has data
                                results_manager.add_arts_introspection_run(
                                    example_name=example_name, 
                                    problem_size=prob_size, 
                                    threads=common_threads, 
                                    aggregated_counters=aggregated_counters_list,
                                    raw_file_details=raw_details_list
                                )
                            else: logger.warning(f"No counter data for {example_name}, Size: {prob_size}, Threads: {common_threads}")
                        else: logger.error(f"ARTS Inspector failed for {example_name}, Size: {prob_size}, Threads: {common_threads}. No counters.")
                        config_manager.configure_arts_for_introspection(enable=False)
                    else: logger.warning(f"ARTS inspector not found: {inspector_exec}. Skipping.")
    
    logger.info("\n--- Aggregating and Reporting Results ---")
    aggregated_data = results_manager.prepare_aggregated_data()
    results_manager.write_csv_report(aggregated_data, output_dir / f"{args.output_prefix}_summary.csv")
    overall_stats = results_manager.compute_overall_statistics(aggregated_data)
    failure_report = results_manager.collect_failure_report(aggregated_data)

    final_json = {
        "script_metadata": {"script_name": Path(__file__).name, "execution_timestamp": datetime.datetime.now().isoformat()},
        "system_information": sys_info_manager.get_system_info(),
        "dependency_versions": sys_info_manager.get_dependency_versions(project_root),
        "reproducibility_info": sys_info_manager.get_reproducibility_info(project_root),
        "command_line_args": vars(args),
        "overall_statistics": overall_stats,
        "failure_summary": failure_report,
        "detailed_aggregated_results": aggregated_data,
        "raw_benchmark_runs": results_manager.get_raw_benchmark_runs(),
        "raw_profiling_runs": results_manager.get_raw_profiling_runs(),
        "arts_introspection_results": results_manager.get_arts_introspection_runs(), 
    }
    results_manager.write_json_report(final_json, output_dir / f"{args.output_prefix}_full_data.json")

    logger.info("\nOverall Statistics Highlights:")
    logger.info(f"  Geom. Mean Speedup (OMP/CARTS): {safe_fmt(overall_stats.get('geometric_mean_speedup'), '.3f')}")
    logger.info(f"  CARTS Faster: {overall_stats.get('carts_faster_configs')}, OMP Faster: {overall_stats.get('omp_faster_configs')}, Tied: {overall_stats.get('tied_configs')}")
    logger.info(f"\nScript finished. Output in {output_dir.resolve()}")
    logger.info("- END CARTS Performance and Introspection Script -")

if __name__ == "__main__":
    main()
