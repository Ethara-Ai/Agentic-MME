#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# LiteLLM-based general mode runner for Agentic-MME
# Supports any LiteLLM-compatible model including AWS Bedrock ARNs
# Usage: python general/run_general_script_litellm.py --model bedrock/converse/<ARN> --task_dir ...

from __future__ import annotations
import os
import json
import argparse
import random
import re
import subprocess
import sys
import time
import traceback
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import litellm

# Add parent directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from common_utils import ensure_dir, image_to_data_url, read_json, safe_name, utc_ts, write_json, get_adaptive_image_params
from dataset_utils import resolve_dataset_root, resolve_image_path
from search_tools import SearchTools, load_search_config
from ast_ops import infer_ops_and_saves, infer_tool_events


def _region_from_arn(arn: str) -> str:
    parts = arn.split(":")
    if len(parts) >= 4 and parts[3]:
        return parts[3]
    return "us-east-1"


_CODE_RE = re.compile(r"<code>(.*?)</code>", re.IGNORECASE | re.DOTALL)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
_ANSWER_UNCLOSED_RE = re.compile(r"<answer>(.*)", re.IGNORECASE | re.DOTALL)  # For unclosed <answer> tags
_TOOL_LOG_RE = re.compile(r"<tool_log>(.*?)</tool_log>", re.IGNORECASE | re.DOTALL)
_THINKING_RE = re.compile(r"<think(?:ing)?>(.*?)</think(?:ing)?>", re.IGNORECASE | re.DOTALL)  # Support both <think> and <thinking>


# ============================================================================
# Tool Definitions for OpenAI Function Calling (Search tools only)
# ============================================================================

SEARCH_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "google_search",
            "description": "Search the web using Google via Serper.dev API. Use for facts, current information, specifications, prices, or any knowledge queries.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query."},
                    "gl": {"type": "string", "description": "Geo location code (e.g., 'us', 'cn'). Default: 'us'"},
                    "hl": {"type": "string", "description": "Language code (e.g., 'en', 'zh'). Default: 'en'"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "google_lens_search",
            "description": "Reverse image search using Google Lens via Serper.dev API. Use to identify objects, brands, logos, landmarks, products, or text in images.",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_ref": {
                        "type": "string",
                        "description": "Quick reference: 'current' for the latest processed image, 'original' for the input image.",
                        "enum": ["current", "original"],
                    },
                    "image_path": {"type": "string", "description": "Filename or full path to a specific image. After code execution, you'll receive a list of generated filenames (e.g., 'transformed_image_0.png'). Use just the filename here to search that specific image."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_webpage",
            "description": "Fetch and read the content of a webpage. Returns clean text extracted from the URL via Jina Reader.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The webpage URL to fetch (must be http/https)."},
                    "max_chars": {"type": "integer", "description": "Maximum characters to return. Default: 12000"},
                },
                "required": ["url"],
            },
        },
    },
    # {
    #     "type": "function",
    #     "function": {
    #         "name": "download_image",
    #         "description": "Download image from URL (use thumbnailUrl from search results). Max 5 images per task, 1-2 per call.",
    #         "parameters": {
    #             "type": "object",
    #             "properties": {
    #                 "url": {"type": "string", "description": "Image URL (use thumbnailUrl from search results)."},
    #             },
    #             "required": ["url"],
    #         },
    #     },
    # },
]

# Maximum number of images that can be downloaded per task
MAX_DOWNLOAD_IMAGES = 5


# ============================================================================
# System Prompts
# ============================================================================

REACT_SYSTEM_PROMPT = r'''You are a multimodal reasoning agent that solves visual questions step by step.

You have access to:
1. **Search tools** (via function calling): google_search, google_lens_search, fetch_webpage
2. **Code execution**: Write Python code in <code> blocks for image manipulation and analysis

## Image Management
- Images are tracked by index: Image 0 is the original input, Images 1, 2, ... are processed results
- Image N corresponds to transformed_image_N.png (e.g., Image 1 = transformed_image_1.png)
- After your code runs, new images will be shown with their index (e.g., "[Image 1: transformed_image_1.png]")
- You can reference any image by its index when using search tools

## Workflow (ReAct Pattern)

For each step:
1. **Think**: Analyze what you know and what you need
2. **Act**: Use search tools OR write code as needed
3. **Observe**: Review results (code output and processed images will be shown with their indices)
4. **Repeat** until you have enough information
5. **Answer**: Provide your final answer

## Response Format

Use these XML blocks as needed (all are OPTIONAL):

<think>
Your reasoning process. Analyze the image, plan your approach, interpret tool results.
</think>

<code>
Python code for image processing. 

Available paths (via environment variables):
- os.environ['ORIGINAL_IMAGE_PATH']: Path to the original input image (Image 0)
- os.environ['PROCESSED_IMAGE_SAVE_PATH']: Directory to save processed images

Naming convention for saved images:
- Save as: transformed_image_1.png, transformed_image_2.png, etc. (starting from 1)
- Full path: os.path.join(os.environ['PROCESSED_IMAGE_SAVE_PATH'], 'transformed_image_1.png')
- Image N corresponds to transformed_image_N.png

You can read any previously saved image from the output directory, including downloaded images (downloaded_image_N.png).
Libraries available: PIL, cv2, numpy, matplotlib, scipy
Use print() to output values. Do NOT use display() or plt.show().
</code>

<answer>
Your final answer. Only include when you have enough information.
</answer>

## Critical Rules

1. **Do NOT combine action and answer in the same turn**: 
   - If you use <code> or call a search tool, do NOT include <answer> in the same response
   - Wait for the results before providing your answer
   - <answer> should only appear when you are ready to give the final answer with NO more actions needed

2. **Image feedback**: After your code runs, you will automatically receive:
   - The stdout/stderr output
   - New images with their indices (e.g., "[Image 1: transformed_image_1.png]")
   - All newly generated images displayed directly

3. **Using specific images with search tools**: 
   - Use google_lens_search with "image_path" parameter to search a specific image
   - Example: {"image_path": "transformed_image_1.png"} to search Image 1
   - Or use "image_ref": "original" for Image 0, "current" for the latest image

# 4. **Downloading images from web**: 
#    - Use download_image to fetch images from URLs found in search results
#    - Downloaded images are saved as downloaded_image_N.png and shown to you
#    - You can then crop/process them with <code> blocks

## Important

- Search tools are called via function calling, NOT in <code>
- Code in <code> blocks will be executed locally
- Think step by step in <think>
- Only provide <answer> when confident and after observing all results
'''



def parse_model_output(text: str) -> Tuple[str, str, str, List[Dict[str, Any]], List[str]]:
    """Return (thinking, python_code, final_answer, tool_log, warnings)."""
    warnings: List[str] = []

    thinking_m = _THINKING_RE.search(text or "")
    code_m = _CODE_RE.search(text or "")
    ans_m = _ANSWER_RE.search(text or "")
    log_m = _TOOL_LOG_RE.search(text or "")

    thinking = (thinking_m.group(1) if thinking_m else "").strip()
    code = (code_m.group(1) if code_m else "").strip()
    
    # Clean markdown code blocks from extracted code
    # Models sometimes add ```python markers inside <code> blocks
    if code:
        # Remove ```python or ``` at the start
        code = re.sub(r'^```python\s*\n?', '', code)
        code = re.sub(r'^```\s*\n?', '', code)
        # Remove ``` at the end
        code = re.sub(r'\n?```\s*$', '', code)
        code = code.strip()
    
    # Try to extract answer with multiple fallback strategies
    ans = ""
    if ans_m:
        # Case 1: Properly closed <answer>...</answer>
        ans = ans_m.group(1).strip()
    else:
        # Case 2: Unclosed <answer> tag (model forgot to close it)
        unclosed_m = _ANSWER_UNCLOSED_RE.search(text or "")
        if unclosed_m:
            ans = unclosed_m.group(1).strip()
            warnings.append("Found unclosed <answer> tag, extracted content anyway")
        elif not code_m and not log_m and not thinking_m:
            # Case 3: No XML tags at all - treat entire response as answer
            # (only if there's no code, tool_log, or thinking)
            clean_text = (text or "").strip()
            if clean_text:
                ans = clean_text
                warnings.append("No <answer> tag found, using entire response as answer")

    tool_log: List[Dict[str, Any]] = []
    if log_m:
        raw = (log_m.group(1) or "").strip()
        if raw:
            try:
                obj = json.loads(raw)
                if isinstance(obj, list):
                    # Flatten if model mistakenly outputs [[...]] instead of [...]
                    if len(obj) == 1 and isinstance(obj[0], list):
                        obj = obj[0]
                    
                    tool_log = [x for x in obj if isinstance(x, dict)]
                    if len(tool_log) != len(obj):
                        non_dict_count = len(obj) - len(tool_log)
                        warnings.append(f"Some tool_log items ({non_dict_count}) were not JSON objects and were ignored.")
                else:
                    warnings.append("tool_log must be a JSON array; got non-array. Ignored.")
            except Exception as e:
                warnings.append(f"Failed to parse tool_log JSON: {e}")
    # Note: tool_log is now optional, no warning if missing

    return thinking, code, ans, tool_log, warnings


def list_transformed_images(tool_images_dir: Path) -> List[Path]:
    """List transformed_image_*.png files in order."""
    if not tool_images_dir.exists():
        return []
    imgs = [p for p in tool_images_dir.iterdir() if p.is_file() and re.match(r"^transformed_image_\d+\.png$", p.name, re.I)]
    def key(p: Path) -> int:
        m = re.search(r"(\d+)", p.stem)
        return int(m.group(1)) if m else 0
    return sorted(imgs, key=key)


def list_all_output_images(tool_images_dir: Path) -> List[Path]:
    """List all image files in tool_images_dir, sorted by modification time."""
    if not tool_images_dir.exists():
        return []
    img_exts = {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.tif'}
    imgs = [p for p in tool_images_dir.iterdir() if p.is_file() and p.suffix.lower() in img_exts]
    # Sort by modification time (newest last)
    return sorted(imgs, key=lambda p: p.stat().st_mtime)


def exec_python_file(
    python_exe: str,
    script_path: Path,
    cwd: Path,
    env: Dict[str, str],
    stdout_path: Path,
    stderr_path: Path,
    timeout_s: int = 180,
) -> Dict[str, Any]:
    """Execute a python file in a subprocess and capture stdout/stderr."""
    cmd = [python_exe, script_path.name]
    p = subprocess.run(
        cmd,
        cwd=str(cwd),
        # Merge with the parent's environment so imports and runtime settings work normally.
        env={**os.environ, **env},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout_s,
        text=True,
    )
    stdout_path.write_text(p.stdout or "", encoding="utf-8")
    stderr_path.write_text(p.stderr or "", encoding="utf-8")
    return {
        "ok": "true" if p.returncode == 0 else "false",
        "returncode": p.returncode,
        "stdout": p.stdout or "",
        "stderr": p.stderr or "",
        "script_path": str(script_path),
    }


def build_tool_use_list_from_code_and_outputs(code: str, tool_images_dir: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Build tool_use_list entries aligned to the atomic toolbox.

    NEW:
    - Use infer_tool_events(code) so UI can show every op occurrence (expanded in loops).
    - Keep infer_ops_and_saves for summary stats (num_saves etc.).
    """
    # Keep old outputs for summary
    ops, saves = infer_ops_and_saves(code)

    # NEW: per-op events for UI
    tool_events = infer_tool_events(code)

    # Collect images in tool_images_dir
    all_images: Dict[str, Path] = {}
    if tool_images_dir.exists():
        for p in tool_images_dir.iterdir():
            if p.is_file() and p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.tif'):
                all_images[p.name] = p

    entries: List[Dict[str, Any]] = []
    prim_hist: Dict[str, int] = {}

    idx = 0
    for ev in tool_events:
        prim = ev.get("tool_name", "unknown")
        args = ev.get("arguments", {}) or {}
        save_name = str(args.get("save") or "")

        # normalize filename only
        if '/' in save_name or '\\' in save_name:
            save_name = save_name.split('/')[-1].split('\\')[-1]

        prim_hist[prim] = prim_hist.get(prim, 0) + 1

        p = all_images.get(save_name) if save_name else None
        out_paths: List[str] = []
        last_path = ""
        if p and p.exists():
            out_paths = [str(p)]
            last_path = str(p)

        entries.append(
            {
                "index": idx,
                "tool_name": prim,
                "raw_tool_name": "python_image_processing",
                "arguments": {
                    "op": prim,
                    "save": save_name,
                    "line": int(args.get("line") or 0),
                    # infer_tool_events 不含 is_standard_name，这里可选：
                    # "is_standard_name": True/False  (如果你需要，可以用 _IMG_RE 判断)
                },
                "output": {
                    "ok": "true" if out_paths else "false",
                    "output_path": last_path,
                    "output_paths": out_paths,
                },
                "timestamp": None,
            }
        )
        idx += 1

    summary = {
        # 你原来统计的是 len(ops)，现在 ops 已经是"展开后的 op 数量"，更符合 UI 需求
        "num_detected_ops": len(ops),
        # NEW: prim_hist 现在是"所有 op 的直方图"，不再是"每个 save 的主操作"
        "primitive_op_hist": prim_hist,
        "num_saves": len(saves),
        "num_images_found": len(all_images),
    }
    return entries, summary


def write_replay_script(run_dir: Path) -> None:
    """Create a minimal replay script that re-executes model_code.py with the same env."""
    script = """#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import runpy
from pathlib import Path

RUN_DIR = Path(__file__).resolve().parent
CODE = RUN_DIR / 'model_code.py'
ORIG = RUN_DIR / 'orig.png'
OUT = RUN_DIR / 'tool_images'

os.environ['ORIGINAL_IMAGE_PATH'] = str(ORIG)
os.environ['LOCAL_INPUT_IMAGE_PATH'] = str(ORIG)
os.environ['PROCESSED_IMAGE_SAVE_PATH'] = str(OUT)
OUT.mkdir(parents=True, exist_ok=True)

runpy.run_path(str(CODE), run_name='__main__')
print('Done. Outputs in:', OUT)
"""
    p = run_dir / "replay_general.py"
    p.write_text(script, encoding="utf-8")



def _normalize_tool_request(entry: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """Return (tool_name, arguments) from a tool_log entry."""
    name = str(entry.get("tool_name") or entry.get("tool") or entry.get("name") or "").strip()
    if not name:
        return "", {}
    args = entry.get("arguments")
    if isinstance(args, dict):
        return name, args
    # fallback: treat other keys as arguments
    args2 = {k: v for k, v in entry.items() if k not in {"tool_name", "tool", "name"}}
    return name, args2


_ENV_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)\}")

def _expand_placeholders(obj: Any, env_map: Dict[str, str]) -> Any:
    """Recursively expand ${VARNAME} placeholders in strings."""
    if isinstance(obj, str):
        def _rep(m: re.Match) -> str:
            key = m.group(1)
            return env_map.get(key, m.group(0))
        return _ENV_PATTERN.sub(_rep, obj)
    if isinstance(obj, list):
        return [_expand_placeholders(x, env_map) for x in obj]
    if isinstance(obj, dict):
        return {k: _expand_placeholders(v, env_map) for k, v in obj.items()}
    return obj



def _execute_tool(
    tool_name: str,
    tool_args: Dict[str, Any],
    search_tools: Optional[SearchTools],
    current_img_path: Path,
    orig_copy: Path,
    tool_images_dir: Path,
    base_env: Dict[str, str],
    python_exe: str,
    run_dir: Path,
    turn: int,
    enable_search: bool,
) -> Dict[str, Any]:
    """Execute a single search tool and return the result."""
    
    try:
        if tool_name == "google_search":
            if not enable_search or search_tools is None:
                return {"ok": False, "error": "Search is not enabled"}
            
            query = str(tool_args.get("query") or "")
            if not query:
                return {"ok": False, "error": "google_search requires 'query' argument"}
            
            return search_tools.google_search(
                query=query,
                gl=tool_args.get("gl"),
                hl=tool_args.get("hl"),
            )
        
        elif tool_name == "google_lens_search":
            if not enable_search or search_tools is None:
                return {"ok": False, "error": "Search is not enabled"}
            
            image_ref = str(tool_args.get("image_ref") or "").strip().lower()
            image_path = tool_args.get("image_path")
            
            if image_ref in {"current", "cur", "latest"}:
                image_path = str(current_img_path)
            elif image_ref in {"orig", "original", "input"}:
                image_path = str(orig_copy.resolve())
            elif image_path:
                # If image_path is just a filename (not a full path), look in tool_images_dir
                from pathlib import Path
                p = Path(image_path)
                if not p.is_absolute() and not p.exists():
                    # Try to find it in tool_images_dir
                    candidate = tool_images_dir / image_path
                    if candidate.exists():
                        image_path = str(candidate.resolve())
            else:
                # Default to original image if nothing specified
                image_path = str(orig_copy.resolve())
            
            return search_tools.google_lens_search(
                image_path=str(image_path) if image_path else None,
            )
        
        elif tool_name == "fetch_webpage":
            if not enable_search or search_tools is None:
                return {"ok": False, "error": "Search is not enabled"}
            
            url = str(tool_args.get("url") or "")
            if not url:
                return {"ok": False, "error": "fetch_webpage requires 'url' argument"}
            
            max_chars = int(tool_args.get("max_chars", 12000) or 12000)
            return search_tools.fetch_webpage(url=url, max_chars=max_chars)
        
        # elif tool_name == "download_image":
        #     # Download image from URL and save to tool_images_dir
        #     url = str(tool_args.get("url") or "")
        #     if not url:
        #         return {"ok": False, "error": "download_image requires 'url' argument"}
        #     
        #     from search_tools import download_image_from_url
        #     result = download_image_from_url(
        #         url=url,
        #         save_dir=str(tool_images_dir),
        #         timeout_s=30,
        #     )
        #     
        #     return result
        
        else:
            return {"ok": False, "error": f"Unknown tool: {tool_name}"}
    
    except Exception as e:
        # Provide more helpful error messages for common network issues
        error_msg = str(e)
        lower_msg = error_msg.lower()
        if "not enough credits" in lower_msg:
            return {
                "ok": False,
                "error": (
                    "Serper API credits exhausted (Not enough credits). "
                    "Please top up Serper credits or disable search for this run."
                ),
            }
        if "504" in error_msg or "timeout" in lower_msg:
            return {"ok": False, "error": "Search service timeout. Please try again or use a different approach."}
        elif "SSL" in error_msg or "ssl" in lower_msg:
            return {"ok": False, "error": "Network error accessing the URL. The website may be unavailable."}
        else:
            return {"ok": False, "error": f"Search failed: {error_msg[:200]}"}


def _short_model_name(model: str) -> str:
    """Create a short directory-friendly model name from potentially long ARN."""
    if "inference-profile/" in model:
        return model.split("inference-profile/")[-1][:40]
    if "provisioned-model/" in model:
        return model.split("provisioned-model/")[-1][:40]
    # Strip bedrock/ prefix if present
    name = model.replace("bedrock/", "")
    return name.replace("/", "_").replace(":", "_")[:60]


class TaskRateLimiter:
    """Sliding-window rate limiter for task execution."""

    def __init__(self, tasks_per_minute: int = 0):
        self._limit = tasks_per_minute
        self._window: deque = deque(maxlen=tasks_per_minute if tasks_per_minute > 0 else None)

    def wait_if_needed(self) -> None:
        if self._limit <= 0:
            return
        if len(self._window) < self._limit:
            return
        oldest = self._window[0]
        elapsed = time.monotonic() - oldest
        if elapsed < 60.0:
            wait = 60.0 - elapsed + 0.1
            print(f"  [THROTTLE] Rate limit: waiting {wait:.1f}s (window: {self._limit} tasks/min)")
            time.sleep(wait)

    def record_task(self) -> None:
        if self._limit <= 0:
            return
        self._window.append(time.monotonic())


_PAYLOAD_DEGRADE_LEVELS = [
    {"max_pixels": 1024 * 1024, "quality": 60, "max_size_mb": 2.0},
    {"max_pixels": 768 * 768,   "quality": 40, "max_size_mb": 1.0},
    {"max_pixels": 512 * 512,   "quality": 25, "max_size_mb": 0.5},
]


def _recompress_messages(messages: List[Dict[str, Any]], level: int, image_list: List[Dict[str, Any]]) -> None:
    params = _PAYLOAD_DEGRADE_LEVELS[level]
    img_idx = 0
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image_url":
                url = part["image_url"]["url"]
                if not url.startswith("data:image"):
                    continue
                if img_idx < len(image_list):
                    src_path = Path(image_list[img_idx]["path"])
                    if src_path.exists():
                        part["image_url"]["url"] = image_to_data_url(
                            src_path,
                            max_pixels=params["max_pixels"],
                            quality=params["quality"],
                            max_size_mb=params["max_size_mb"],
                        )
                img_idx += 1


def _completion_with_payload_retry(
    api_kwargs: Dict[str, Any],
    messages: List[Dict[str, Any]],
    image_list: List[Dict[str, Any]],
    timeout: int = 300,
    max_timeout_retries: int = 3,
) -> Any:
    api_kwargs.setdefault("timeout", timeout)
    
    for level in range(-1, len(_PAYLOAD_DEGRADE_LEVELS)):
        for timeout_attempt in range(max_timeout_retries + 1):
            try:
                return litellm.completion(**api_kwargs)
            except (litellm.Timeout, litellm.APIConnectionError) as e:
                if timeout_attempt >= max_timeout_retries:
                    raise
                wait = min(30, 5 * (2 ** timeout_attempt)) + random.uniform(0, 2)
                print(f"  [TIMEOUT] Attempt {timeout_attempt + 1}/{max_timeout_retries} — "
                      f"retrying in {wait:.1f}s: {type(e).__name__}")
                time.sleep(wait)
            except litellm.BadRequestError as e:
                err_msg = str(e).lower()
                if "length limit exceeded" not in err_msg and "payload too large" not in err_msg:
                    raise
                next_level = level + 1
                if next_level >= len(_PAYLOAD_DEGRADE_LEVELS):
                    raise
                params = _PAYLOAD_DEGRADE_LEVELS[next_level]
                print(f"  [RETRY] Payload too large — recompressing images (level {next_level}: "
                      f"quality={params['quality']}, max_mb={params['max_size_mb']})")
                _recompress_messages(messages, next_level, image_list)
                break
    api_kwargs["timeout"] = timeout
    return litellm.completion(**api_kwargs)


def run_one_rollout(
    task_json: Path,
    dataset_root: Path,
    images_dir: Optional[Path],
    out_dir: Path,
    model: str,
    temperature: float,
    python_exe: str,
    enable_search: bool = False,
    search_cfg_path: Optional[Path] = None,
    max_rounds: int = 15,
    max_tool_calls: int = 15,
    max_image_pixels: int = 2048 * 2048,
    image_quality: int = 95,
    use_native_tools: bool = True,
    api_key: str = "",
    cost_base_model: str = "",
    api_timeout: int = 300,
) -> Dict[str, Any]:
    """Multi-turn runner that executes tools and feeds results back to the model.
    
    Args:
        use_native_tools: If True, use OpenAI native function calling (tools= parameter).
                         If False, use legacy XML-based tool_log parsing.
    """

    task_cfg = read_json(task_json)
    img_path_result = resolve_image_path(task_json, task_cfg, dataset_root, images_dir)
    
    # Handle single or multiple images
    if isinstance(img_path_result, list):
        img_paths = img_path_result
    else:
        img_paths = [img_path_result]

    item_key = safe_name(task_json.stem)
    run_dir = ensure_dir(out_dir / item_key)
    tool_images_dir = ensure_dir(run_dir / "tool_images")

    # copy orig image(s)
    from PIL import Image
    orig_copies: List[Path] = []
    for i, img_path in enumerate(img_paths):
        if len(img_paths) == 1:
            orig_copy = run_dir / "orig.png"
        else:
            orig_copy = run_dir / f"orig_{i}.png"
        Image.open(img_path).save(orig_copy, "PNG")
        orig_copies.append(orig_copy)
    
    # Primary original image (for backward compatibility)
    orig_copy = orig_copies[0]

    prompt = (task_cfg.get("input") or {}).get("prompt", "")

    # Search tools
    search_tools: Optional[SearchTools] = None
    if enable_search:
        cfg = load_search_config(str(search_cfg_path) if search_cfg_path else None)
        # Default: keep cache under run_dir for replayability.
        if not getattr(cfg, "cache_dir", None):
            cfg.cache_dir = str(ensure_dir(run_dir / "search_cache"))
        # Pass task_id for organized cache naming: _search_cache/{task_id}/serper_search_1.json
        task_id = task_cfg.get("task_id", "") or run_dir.name
        search_tools = SearchTools(cfg, task_id=task_id)

    # Runtime env map for placeholder expansion + code execution
    import os as _os
    base_env = dict(_os.environ)
    base_env["PROCESSED_IMAGE_SAVE_PATH"] = str(tool_images_dir.resolve())

    current_img_path = orig_copy.resolve()
    
    # Image index tracking: list of (path, label) tuples, 0-indexed
    # Index 0 = original image (or first image if multiple), Index N = transformed_image_N.png
    image_list: List[Dict[str, Any]] = []
    download_count = 0  # Track number of downloaded images (max MAX_DOWNLOAD_IMAGES per task)
    for i, oc in enumerate(orig_copies):
        if len(orig_copies) == 1:
            image_list.append({"index": 0, "path": str(oc), "label": "original input image"})
        else:
            image_list.append({"index": i, "path": str(oc), "label": f"original input image {i+1}"})

    # Build tools list for native function calling (search tools always use function calling)
    tools_list = []
    if enable_search:
        tools_list.extend(SEARCH_TOOLS_SCHEMA)

    # Select system prompt based on mode
    # use_native_tools controls visual tools (toolbox mode), not search tools
    system_prompt = REACT_SYSTEM_PROMPT

    # Build initial user message with all input images
    # Use max_size_mb to ensure original images don't exceed API limits
    user_content: List[Dict[str, Any]] = []
    for i, oc in enumerate(orig_copies):
        user_content.append({"type": "image_url", "image_url": {"url": image_to_data_url(oc, max_image_pixels, image_quality, max_size_mb=8.0)}})
    
    if len(orig_copies) == 1:
        user_content.append({"type": "text", "text": f"[Image 0: original input]\n\n{prompt}"})
        history_text = f"[Image 0: original input - {img_paths[0].name}]\n\n{prompt}"
    else:
        img_labels = ", ".join([f"Image {i}" for i in range(len(orig_copies))])
        user_content.append({"type": "text", "text": f"[{img_labels}: original inputs]\n\n{prompt}"})
        history_text = f"[{img_labels}: original inputs - {', '.join([p.name for p in img_paths])}]\n\n{prompt}"

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    
    # Track conversation history for JSON export (without base64 image data)
    conversation_history: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": history_text},
    ]

    usage = {"api_calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "total_cost_usd": 0.0}
    all_warnings: List[str] = []
    tool_call_events: List[Dict[str, Any]] = []
    exec_events: List[Dict[str, Any]] = []
    code_turn_paths: List[Path] = []

    final_answer = ""
    tool_calls_total = 0
    last_turn_had_code = False  # Track if last turn executed code
    tool_budget_exhausted = False  # Stop exposing tool calling after budget is consumed
    serper_credit_warned = False  # Avoid duplicate warnings for same root cause

    def _request_final_answer(reason: str) -> None:
        prompt_text = (
            f"{reason} Please provide your final answer now with NO additional tool calls "
            f"and NO additional <code> blocks. Use <answer>...</answer> format."
        )
        messages.append({"role": "user", "content": prompt_text})
        conversation_history.append({"role": "user", "content": f"[System request] {prompt_text}"})

    def _maybe_warn_serper_credit(tool_name: str, out: Any, turn_idx: int) -> None:
        nonlocal serper_credit_warned
        if serper_credit_warned:
            return
        err_text = ""
        if isinstance(out, dict):
            v = out.get("error")
            if isinstance(v, str):
                err_text = v
        elif isinstance(out, str):
            err_text = out
        if "not enough credits" in err_text.lower():
            serper_credit_warned = True
            all_warnings.append(
                f"[runtime] Serper API credits exhausted (turn {turn_idx}, tool={tool_name}). "
                f"Search quality is degraded; please top up Serper credits."
            )

    for turn in range(max_rounds):
        # Build API call kwargs
        api_kwargs = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 12000,
            "api_key": api_key or None,
            "aws_region_name": _region_from_arn(model),
        }
        # Search tools always use function calling (regardless of use_native_tools)
        if tools_list and not tool_budget_exhausted:
            api_kwargs["tools"] = tools_list
            api_kwargs["tool_choice"] = "auto"
        
        resp = _completion_with_payload_retry(api_kwargs, messages, image_list, timeout=api_timeout, max_timeout_retries=10)
        usage["api_calls"] += 1
        step_cost = 0.0
        if getattr(resp, "usage", None):
            usage["prompt_tokens"] += getattr(resp.usage, "prompt_tokens", 0) or 0
            usage["completion_tokens"] += getattr(resp.usage, "completion_tokens", 0) or 0
            usage["total_tokens"] += getattr(resp.usage, "total_tokens", 0) or 0
        try:
            step_cost = litellm.completion_cost(completion_response=resp, base_model=cost_base_model or None)
            usage["total_cost_usd"] += step_cost
        except Exception:
            pass
        pt = getattr(resp.usage, "prompt_tokens", 0) or 0
        ct = getattr(resp.usage, "completion_tokens", 0) or 0
        print(f"  [COST] turn={turn} | {pt} prompt + {ct} completion | ${step_cost:.6f}")

        message = resp.choices[0].message
        raw = message.content or ""
        
        # Save raw model output - include function calls if present
        raw_output_parts = []
        if raw:
            raw_output_parts.append(raw)
        
        # Also save function call information if present
        tool_calls_from_api = getattr(message, "tool_calls", None) or []
        if tool_calls_from_api:
            raw_output_parts.append("\n--- Function Calls ---")
            for tc in tool_calls_from_api:
                raw_output_parts.append(f"Tool: {tc.function.name}")
                raw_output_parts.append(f"Arguments: {tc.function.arguments}")
        
        (run_dir / f"raw_model_output_turn_{turn}.txt").write_text("\n".join(raw_output_parts), encoding="utf-8")
        
        last_turn_had_code = False  # Reset for this turn
        
        # Always parse text content for <thinking>, <code>, <answer> blocks
        thinking, code, ans, tool_log, parse_warnings = parse_model_output(raw)
        all_warnings.extend([f"[turn {turn}] {w}" for w in parse_warnings])
        
        # Execute <code> block if present (regardless of function calls)
        if code.strip():
            last_turn_had_code = True
            code_path = run_dir / f"model_code_turn_{turn}.py"
            code_path.write_text(code, encoding="utf-8")
            code_turn_paths.append(code_path)

            exec_stdout = run_dir / f"exec_stdout_turn_{turn}.txt"
            exec_stderr = run_dir / f"exec_stderr_turn_{turn}.txt"

            # Record existing images before execution
            imgs_before = set(p.name for p in list_all_output_images(tool_images_dir)) if tool_images_dir.exists() else set()

            env = dict(base_env)
            # Provide both original image path and output directory
            env["ORIGINAL_IMAGE_PATH"] = str(orig_copy.resolve())
            # Keep LOCAL_INPUT_IMAGE_PATH for backward compatibility (points to latest image)
            env["LOCAL_INPUT_IMAGE_PATH"] = str(current_img_path)

            exec_out = exec_python_file(
                python_exe=python_exe,
                script_path=code_path,
                cwd=run_dir,
                env=env,
                stdout_path=exec_stdout,
                stderr_path=exec_stderr,
                timeout_s=180,
            )
            exec_events.append(
                {
                    "tool_name": "run_visual_processing_code",
                    "raw_tool_name": "script_exec",
                    "arguments": {"python_exe": python_exe, "turn": turn},
                    "output": exec_out,
                }
            )

            # Find newly generated images from this turn
            imgs_after = set(p.name for p in list_all_output_images(tool_images_dir)) if tool_images_dir.exists() else set()
            new_img_names = imgs_after - imgs_before
            new_imgs = [p for p in list_all_output_images(tool_images_dir) if p.name in new_img_names]
            # Sort by modification time
            new_imgs.sort(key=lambda p: p.stat().st_mtime)
            
            # Add new images to image_list with indices
            # Extract index from filename: transformed_image_N.png -> index N
            new_img_indices = []
            for img_p in new_imgs:
                match = re.match(r"transformed_image_(\d+)\.png", img_p.name, re.I)
                if match:
                    new_idx = int(match.group(1))
                else:
                    # Fallback: use sequential index
                    new_idx = len(image_list)
                image_list.append({"index": new_idx, "path": str(img_p), "label": f"generated in turn {turn}"})
                new_img_indices.append((new_idx, img_p))
            
            # Update current image to the latest output image if available
            imgs = list_transformed_images(tool_images_dir)
            if not imgs:
                imgs = list_all_output_images(tool_images_dir)
            if imgs:
                current_img_path = imgs[-1].resolve()
            
            # Build code execution result message
            code_result_parts = []
            if exec_out.get("ok") == "true":
                stdout_content = (run_dir / f"exec_stdout_turn_{turn}.txt").read_text(encoding="utf-8", errors="replace")[:2000]
                if stdout_content.strip():
                    code_result_parts.append(f"[Code output]\n{stdout_content}")
                else:
                    code_result_parts.append("[Code executed successfully, no output]")
            else:
                stderr_content = (run_dir / f"exec_stderr_turn_{turn}.txt").read_text(encoding="utf-8", errors="replace")[:2000]
                code_result_parts.append(f"[Code error]\n{stderr_content}")
            
            # List new images generated in this turn with indices
            if new_img_indices:
                img_info_list = [f"Image {idx}: {p.name}" for idx, p in new_img_indices]
                code_result_parts.append(f"[New images: {', '.join(img_info_list)}]")
            
            code_result_msg = "\n".join(code_result_parts)
            
            # If model already provided an answer, we're done
            if ans.strip():
                final_answer = ans.strip()
                break
            
            # Otherwise, feed code execution result back to model for next turn
            messages.append({"role": "assistant", "content": raw})
            
            user_content = []
            # Use adaptive compression based on how many images are already in conversation
            img_count = sum(
                1
                for m in messages
                if isinstance(m.get("content"), list)
                and any(c.get("type") == "image_url" for c in m["content"] if isinstance(c, dict))
            )
            adaptive_params = get_adaptive_image_params(img_count)
            
            for idx_img, img_p in new_img_indices:
                user_content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_to_data_url(
                                img_p,
                                max_pixels=adaptive_params["max_pixels"],
                                quality=adaptive_params["quality"],
                                max_size_mb=adaptive_params["max_size_mb"],
                            )
                        },
                    }
                )
                user_content.append({"type": "text", "text": f"[Image {idx_img}: {img_p.name}]"})
            user_content.append({"type": "text", "text": code_result_msg})
            
            messages.append({"role": "user", "content": user_content})
            
            # Track in conversation history (without base64)
            conversation_history.append({"role": "assistant", "content": raw})
            history_parts = [code_result_msg]
            for idx_img, img_p in new_img_indices:
                history_parts.append(f"[Image {idx_img} shown: {img_p.name}]")
            conversation_history.append({"role": "user", "content": "\n".join(history_parts)})
            continue
        
        if tool_calls_from_api:
            # Native function calling mode - process search tool calls
            messages.append(message.model_dump())
            
            if raw:
                conversation_history.append({"role": "assistant", "content": raw})
            
            downloaded_images_this_turn: List[Tuple[int, Path]] = []
            hit_tool_budget_this_turn = False
            
            for tc in tool_calls_from_api:
                tool_calls_total += 1
                if tool_calls_total > max_tool_calls:
                    all_warnings.append(f"[runtime] Exceeded max_tool_calls={max_tool_calls}")
                    tool_budget_exhausted = True
                    hit_tool_budget_this_turn = True
                    break
                
                func_name = tc.function.name
                try:
                    func_args = json.loads(tc.function.arguments)
                except:
                    func_args = {}
                
                tool_result = _execute_tool(
                    tool_name=func_name,
                    tool_args=func_args,
                    search_tools=search_tools,
                    current_img_path=current_img_path,
                    orig_copy=orig_copy,
                    tool_images_dir=tool_images_dir,
                    base_env=base_env,
                    python_exe=python_exe,
                    run_dir=run_dir,
                    turn=turn,
                    enable_search=enable_search,
                )
                _maybe_warn_serper_credit(func_name, tool_result, turn)
                
                conversation_history.append({
                    "role": "tool_call",
                    "content": json.dumps({"name": func_name, "arguments": func_args}, ensure_ascii=False),
                })
                
                tool_call_events.append({
                    "tool_name": func_name,
                    "arguments": func_args,
                    "output": tool_result,
                    "turn": turn,
                    "index": len(tool_call_events),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
                
                if isinstance(tool_result, dict) and "context" in tool_result:
                    model_content = tool_result["context"]
                elif isinstance(tool_result, dict) and "text" in tool_result:
                    model_content = tool_result["text"]
                elif isinstance(tool_result, dict) and "error" in tool_result:
                    model_content = f"Error: {tool_result['error']}"
                else:
                    model_content = json.dumps(tool_result, ensure_ascii=False)
                
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": model_content,
                })
                
                conversation_history.append({
                    "role": "tool_response",
                    "content": model_content,
                })
            
            if downloaded_images_this_turn:
                img_count = sum(
                    1
                    for m in messages
                    if isinstance(m.get("content"), list)
                    and any(c.get("type") == "image_url" for c in m["content"] if isinstance(c, dict))
                )
                adaptive_params = get_adaptive_image_params(img_count)
                
                user_content = []
                for idx_img, img_p in downloaded_images_this_turn:
                    user_content.append({
                        "type": "image_url",
                        "image_url": {"url": image_to_data_url(
                            img_p,
                            max_pixels=adaptive_params["max_pixels"],
                            quality=adaptive_params["quality"],
                            max_size_mb=adaptive_params["max_size_mb"]
                        )},
                    })
                    user_content.append({"type": "text", "text": f"[Image {idx_img}: {img_p.name} - downloaded from URL]"})
                
                user_content.append({"type": "text", "text": "Above are the downloaded images. You can now analyze them or use <code> blocks to process them further."})
                
                messages.append({"role": "user", "content": user_content})
                
                history_parts = []
                for idx_img, img_p in downloaded_images_this_turn:
                    history_parts.append(f"[Image {idx_img} shown: {img_p.name} - downloaded]")
                history_parts.append("Above are the downloaded images. You can now analyze them or use <code> blocks to process them further.")
                conversation_history.append({"role": "user", "content": "\n".join(history_parts)})
            
            if hit_tool_budget_this_turn:
                if turn < max_rounds - 1:
                    _request_final_answer(
                        f"Tool-call budget has been exhausted ({max_tool_calls})."
                    )
                    continue
                break

            continue
        
        # Process text response (no function calls)
        messages.append({"role": "assistant", "content": raw})
        conversation_history.append({"role": "assistant", "content": raw})

        # Legacy XML tool_log path
        normalized: List[Tuple[str, Dict[str, Any]]] = []
        for entry in tool_log:
            if not isinstance(entry, dict):
                all_warnings.append(f"[turn {turn}] tool_log entry not an object: {type(entry)}")
                continue
            tname, targs = _normalize_tool_request(entry)
            if not tname:
                all_warnings.append(f"[turn {turn}] tool_log entry missing tool_name")
                continue
            normalized.append((tname, targs))

        if normalized:
            if tool_budget_exhausted:
                all_warnings.append("[runtime] Tool-call budget exhausted; skipping tool_log execution.")
                if turn < max_rounds - 1:
                    _request_final_answer("Tool-call budget is exhausted.")
                    continue
                break

            if not enable_search or search_tools is None:
                final_answer = ""
                all_warnings.append("[runtime] enable_search is false but tool_log requested tools.")
                break

            env_map = {
                "LOCAL_INPUT_IMAGE_PATH": str(current_img_path),
                "PROCESSED_IMAGE_SAVE_PATH": str(tool_images_dir.resolve()),
            }

            results_for_model: List[Dict[str, Any]] = []
            hit_tool_budget_this_turn = False
            for tname, targs in normalized:
                tool_calls_total += 1
                if tool_calls_total > max_tool_calls:
                    all_warnings.append(f"[runtime] Exceeded max_tool_calls={max_tool_calls}")
                    tool_budget_exhausted = True
                    hit_tool_budget_this_turn = True
                    break

                targs = _expand_placeholders(targs, env_map)

                try:
                    if tname == "google_search":
                        query = str(targs.get("query") or "")
                        if not query:
                            raise ValueError("google_search requires arguments.query")
                        out = search_tools.google_search(
                            query=query,
                            gl=targs.get("gl"),
                            hl=targs.get("hl"),
                            page=int(targs.get("page", 1) or 1),
                            search_type=str(targs.get("type") or "search"),
                            autocorrect=bool(targs.get("autocorrect", True)),
                        )
                    elif tname == "google_lens_search":
                        image_url = targs.get("image_url")
                        image_path = targs.get("image_path")
                        image_ref = str(targs.get("image_ref") or "").strip().lower()
                        if image_ref in {"current", "cur", "latest"}:
                            image_path = str(current_img_path)
                        elif image_ref in {"orig", "original", "input"}:
                            image_path = str(orig_copy.resolve())

                        out = search_tools.google_lens_search(
                            image_url=str(image_url) if isinstance(image_url, str) and image_url else None,
                            image_path=str(image_path) if isinstance(image_path, str) and image_path else None,
                            page=int(targs.get("page", 1) or 1),
                            num=int(targs.get("num", 10) or 10),
                        )
                    elif tname == "fetch_webpage":
                        url = str(targs.get("url") or "")
                        if not url:
                            raise ValueError("fetch_webpage requires arguments.url")
                        out = search_tools.fetch_webpage(
                            url=url,
                            max_chars=int(targs.get("max_chars", 12000) or 12000),
                        )
                    else:
                        raise ValueError(f"Unknown tool_name: {tname}")
                    _maybe_warn_serper_credit(tname, out, turn)

                    tool_call_events.append(
                        {
                            "tool_name": tname,
                            "raw_tool_name": tname,
                            "arguments": targs,
                            "output": out,
                            "turn": turn,
                        }
                    )
                    results_for_model.append({"tool_name": tname, "arguments": targs, "output": out})

                except Exception as e:
                    err = {"ok": False, "error": str(e)}
                    _maybe_warn_serper_credit(tname, err, turn)
                    tool_call_events.append(
                        {
                            "tool_name": tname,
                            "raw_tool_name": tname,
                            "arguments": targs,
                            "output": err,
                            "turn": turn,
                        }
                    )
                    results_for_model.append({"tool_name": tname, "arguments": targs, "output": err})

            img_count = sum(
                1
                for m in messages
                if isinstance(m.get("content"), list)
                and any(c.get("type") == "image_url" for c in m["content"] if isinstance(c, dict))
            )
            adaptive_params = get_adaptive_image_params(img_count)
            
            results_text = "<tool_results>" + json.dumps(results_for_model, ensure_ascii=False) + "</tool_results>"
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_to_data_url(
                            current_img_path, 
                            max_pixels=adaptive_params["max_pixels"],
                            quality=adaptive_params["quality"],
                            max_size_mb=adaptive_params["max_size_mb"]
                        )}},
                        {"type": "text", "text": results_text},
                    ],
                }
            )
            
            for tname, targs in normalized:
                conversation_history.append({
                    "role": "tool_call",
                    "content": json.dumps({"name": tname, "arguments": targs}, ensure_ascii=False),
                })
            for res in results_for_model:
                out = res.get("output", {})
                if isinstance(out, dict) and "context" in out:
                    resp_content = out["context"]
                elif isinstance(out, dict) and "text" in out:
                    resp_content = out["text"]
                elif isinstance(out, dict) and "error" in out:
                    resp_content = f"Error: {out['error']}"
                else:
                    resp_content = json.dumps(out, ensure_ascii=False)
                conversation_history.append({
                    "role": "tool_response",
                    "content": resp_content,
                })

            if hit_tool_budget_this_turn:
                if turn < max_rounds - 1:
                    _request_final_answer(
                        f"Tool-call budget has been exhausted ({max_tool_calls})."
                    )
                    continue
                break
            continue

        # No tools requested: treat as final answer
        parsed_answer = (ans or "").strip()
        if parsed_answer:
            final_answer = parsed_answer
            break

        raw_text = (raw or "").strip()
        if raw_text:
            # Fallback: keep non-empty plain text so we don't silently write empty answers.
            final_answer = raw_text
            all_warnings.append(f"[turn {turn}] No parseable <answer>; using raw assistant text as final answer")
            break

        all_warnings.append(f"[turn {turn}] Empty assistant response with no tools/code")
        if turn < max_rounds - 1:
            _request_final_answer("Your previous response was empty.")
            continue
        break

    # If we still don't have an answer (including early breaks), request one final answer.
    if not final_answer:
        final_prompt = "Please provide your final answer now based on all the information gathered. Use <answer>your answer</answer> format."
        messages.append({
            "role": "user",
            "content": final_prompt,
        })
        conversation_history.append({"role": "user", "content": final_prompt})
        
        try:
            final_kwargs = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": 2000,
                "api_key": api_key or None,
                "aws_region_name": _region_from_arn(model),
            }
            final_resp = _completion_with_payload_retry(final_kwargs, messages, image_list, timeout=api_timeout, max_timeout_retries=10)
            usage["api_calls"] += 1
            if getattr(final_resp, "usage", None):
                usage["prompt_tokens"] += getattr(final_resp.usage, "prompt_tokens", 0) or 0
                usage["completion_tokens"] += getattr(final_resp.usage, "completion_tokens", 0) or 0
                usage["total_tokens"] += getattr(final_resp.usage, "total_tokens", 0) or 0
            try:
                final_cost = litellm.completion_cost(completion_response=final_resp, base_model=cost_base_model or None)
                usage["total_cost_usd"] += final_cost
            except Exception:
                final_cost = 0.0
            fpt = getattr(final_resp.usage, "prompt_tokens", 0) or 0
            fct = getattr(final_resp.usage, "completion_tokens", 0) or 0
            print(f"  [COST] final | {fpt} prompt + {fct} completion | ${final_cost:.6f}")
            
            final_raw = final_resp.choices[0].message.content or ""
            (run_dir / f"raw_model_output_final.txt").write_text(final_raw, encoding="utf-8")
            conversation_history.append({"role": "assistant", "content": final_raw})
            
            _, _, final_ans, _, _ = parse_model_output(final_raw)
            if final_ans.strip():
                final_answer = final_ans.strip()
            else:
                final_answer = final_raw.strip()
        except Exception as e:
            all_warnings.append(f"[runtime] Failed to get final answer: {e}")

    # Save final answer
    (run_dir / "model_answer.txt").write_text(final_answer, encoding="utf-8")
    
    # Save image index tracking
    write_json(run_dir / "image_index.json", image_list)
    
    # Save conversation history as formatted JSON (readable)
    tools_json = json.dumps(tools_list, ensure_ascii=False) if tools_list else "[]"
    conversation_record = {
        "task_id": task_cfg.get("task_id", ""),
        "tools": tools_json,
        "images": image_list,
        "messages": conversation_history,
        "final_answer": final_answer,
    }
    with open(run_dir / "conversation.json", "w", encoding="utf-8") as f:
        json.dump(conversation_record, f, ensure_ascii=False, indent=2)

    # =====================================================================
    # Build tool_use_list aligned to atomic toolbox
    # IMPORTANT CHANGE (compared to your old version):
    #   - Previously: tool_entries built from all_saves (one entry per save, using s.op_guess)
    #   - Now: tool_entries built from infer_tool_events(code) (one entry per OP occurrence)
    #     so UI can show every crop/enhance/... including loop expansions.
    # =====================================================================
    tool_use_list: List[Dict[str, Any]] = []

    # Collect analysis across all code turns
    total_detected_ops = 0
    total_saves = 0
    total_prim_hist: Dict[str, int] = {}

    # NEW: collect per-op tool events
    all_tool_events: List[Dict[str, Any]] = []

    for code_path in code_turn_paths:
        code_content = code_path.read_text(encoding="utf-8")

        # Summary stats still based on infer_ops_and_saves
        ops, saves = infer_ops_and_saves(code_content)
        total_detected_ops += len(ops)
        total_saves += len(saves)

        # NEW: UI/tool_use_list should reflect every op
        # NOTE: requires `from ast_ops import infer_tool_events`
        tool_events = infer_tool_events(code_content)
        all_tool_events.extend(tool_events)

        # Histogram now counts every op occurrence
        for ev in tool_events:
            prim = ev.get("tool_name", "unknown")
            total_prim_hist[prim] = total_prim_hist.get(prim, 0) + 1

    # Get all images in tool_images_dir
    all_images: Dict[str, Path] = {}
    if tool_images_dir.exists():
        for p in tool_images_dir.iterdir():
            if p.is_file() and p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.tif'):
                all_images[p.name] = p

    # NEW: Build tool entries from per-op tool events (not saves)
    tool_entries: List[Dict[str, Any]] = []
    for idx, ev in enumerate(all_tool_events):
        prim = ev.get("tool_name", "unknown")
        args = ev.get("arguments", {}) or {}
        save_name = str(args.get("save") or "")

        # Extract just filename if it's a path
        if '/' in save_name or '\\' in save_name:
            save_name = save_name.split('/')[-1].split('\\')[-1]

        p = all_images.get(save_name) if save_name else None

        out_paths: List[str] = []
        last_path = ""
        if save_name and p and p.exists():
            out_paths = [str(p)]
            last_path = str(p)

        tool_entries.append({
            "index": idx,
            "tool_name": prim,
            "raw_tool_name": "python_image_processing",
            "arguments": {
                "op": prim,
                "save": save_name,
                "line": int(args.get("line") or 0),
            },
            "output": {
                "ok": "true" if out_paths else "false",
                "output_path": last_path,
                "output_paths": out_paths,
            },
            "timestamp": None,
        })

    tool_summary = {
        "num_detected_ops": total_detected_ops,
        "primitive_op_hist": total_prim_hist,
        "num_saves": total_saves,
        "num_images_found": len(all_images),
    }

    # Count search tool calls separately
    search_tool_hist: Dict[str, int] = {}
    for ev in tool_call_events:
        tname = ev.get("tool_name", "")
        if tname in ("google_search", "google_lens_search", "fetch_webpage"):
            search_tool_hist[tname] = search_tool_hist.get(tname, 0) + 1

    # Add script_exec events (in order)
    combined_entries: List[Dict[str, Any]] = []
    combined_entries.extend(tool_call_events)  # search tools
    combined_entries.extend(tool_entries)      # visual ops (per-op)
    combined_entries.extend(exec_events)       # code exec

    for i, ev in enumerate(combined_entries):
        ev["index"] = i
        ev["timestamp"] = utc_ts()
        tool_use_list.append(ev)

    write_json(run_dir / "tool_use_list.json", tool_use_list)

    # Replay script for multi-turn code execution
    if code_turn_paths:
        replay_lines = [
            "import os, runpy",
            "from pathlib import Path",
            "",
            "RUN_DIR = Path(__file__).resolve().parent",
            "ORIG = RUN_DIR / 'orig.png'",
            "OUT = RUN_DIR / 'tool_images'",
            "OUT.mkdir(parents=True, exist_ok=True)",
            "",
            "# Set up environment variables",
            "os.environ['ORIGINAL_IMAGE_PATH'] = str(ORIG)",
            "os.environ['PROCESSED_IMAGE_SAVE_PATH'] = str(OUT)",
            "",
            "cur = ORIG",
            "",
        ]
        for i, pth in enumerate(code_turn_paths):
            replay_lines += [
                f"print('== TURN {i} ==')",
                "os.environ['LOCAL_INPUT_IMAGE_PATH'] = str(cur)",
                f"runpy.run_path(str(RUN_DIR / '{pth.name}'), run_name='__main__')",
                "# Update cur to latest transformed image if produced",
                "imgs = sorted(OUT.glob('transformed_image_*.png'))",
                "if not imgs:",
                "    imgs = sorted([p for p in OUT.iterdir() if p.suffix.lower() in ('.png', '.jpg', '.jpeg')])",
                "if imgs:",
                "    cur = imgs[-1]",
                "",
            ]
        replay_lines += ["print('Done. Outputs in:', OUT)"]
        (run_dir / "replay_general_rollout.py").write_text("\n".join(replay_lines) + "\n", encoding="utf-8")
    else:
        write_replay_script(run_dir)

    run_meta = {
        "task_id": task_cfg.get("task_id", ""),
        "task_file": str(task_json.resolve()),
        "mode": "general_rollout",
        "driver": "script_rollout_litellm",
        "model": model,
        "temperature": temperature,
        "paths": {
            "run_dir": str(run_dir),
            "orig": str(orig_copy),
            "processed_dir": str(tool_images_dir),
        },
        "usage": usage,
        "effective_tool_calls": len(tool_call_events) + len(tool_entries),  # Search tools + visual ops, exclude code execution events
        "total_images": len(image_list),
        "code_analysis": tool_summary,
        "search_analysis": {
            "search_tool_calls": sum(search_tool_hist.values()),
            "search_tool_hist": search_tool_hist,
        },
        "warnings": all_warnings,
    }
    write_json(run_dir / "run_meta.json", run_meta)
    return run_meta

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task_json", type=str, default="")
    ap.add_argument("--task_dir", type=str, default="")
    ap.add_argument("--dataset_root", type=str, default="")
    ap.add_argument("--images_dir", type=str, default="")
    ap.add_argument("--out_dir", type=str, default="", help="Output directory. Default: runs/general/{model_name}")
    ap.add_argument("--model", type=str, default="")
    ap.add_argument("--temperature", type=float, default=0.0)

    # Multi-turn tool execution
    ap.add_argument("--use_native_tools", action="store_true", help="Use OpenAI native function calling for visual tools (toolbox mode)")
    ap.add_argument("--max_rounds", type=int, default=15, help="Max model turns")
    ap.add_argument("--max_tool_calls", type=int, default=15, help="Max total tool calls across turns")
    
    # Rate limiting
    ap.add_argument("--task_delay", type=float, default=2.0, help="Delay in seconds between tasks (to avoid rate limits)")
    ap.add_argument("--max_retries", type=int, default=3, help="Max retries for rate limit errors")
    ap.add_argument("--tasks_per_minute", type=int, default=0, help="Max tasks per 60s sliding window (0=unlimited)")
    ap.add_argument("--litellm_retries", type=int, default=3, help="LiteLLM SDK internal retry count")
    ap.add_argument("--timeout", type=int, default=300, help="API call timeout in seconds (default 300s/5min)")

    # Web search (Serper.dev)
    ap.add_argument("--enable_search", action="store_true", default=True, help="Enable web search tools (default: enabled)")
    ap.add_argument("--no_search", action="store_true", default=False, help="Disable web search tools")
    ap.add_argument("--search_config", type=str, default="configs/search_config.json", help="Path to search config JSON (serper_api_key, imgbb_api_key, jina_api_key, cache_dir, replay, ...)")

    # LiteLLM API key (used for Bedrock and other providers)
    ap.add_argument("--api_key", type=str, default="", help="API key for the model provider")
    ap.add_argument("--api_config", type=str, default="", help="Path to JSON config with {api_key, model, base_model} (e.g. configs/api.json)")

    # Image size control
    ap.add_argument("--max_image_pixels", type=int, default=2048*2048, 
                    help="Maximum image pixels (width*height). Larger images will be resized. Default: 4194304 (2048x2048)")
    ap.add_argument("--image_quality", type=int, default=95, 
                    help="JPEG quality for resized images (1-100). Default: 95")

    ap.add_argument("--python", type=str, default=sys.executable, help="Python executable used to run model_code.py")
    
    # Skip and shard options
    ap.add_argument("--skip_existing", action="store_true", help="Skip tasks that already have run_meta.json")
    ap.add_argument("--shard", type=int, default=0, help="Shard index (0-based) for parallel runs")
    ap.add_argument("--num_shards", type=int, default=1, help="Total number of shards for parallel runs")
    ap.add_argument("--max_tasks", type=int, default=0, help="Max tasks to process (0 = unlimited)")
    
    args = ap.parse_args()

    api_key = args.api_key
    model = args.model
    base_model = ""

    if args.api_config:
        cfg = read_json(Path(args.api_config))
        api_key = api_key or cfg.get("api_key", "")
        model = model or cfg.get("model", "")
        base_model = cfg.get("base_model", "")

    if api_key:
        os.environ["LITELLM_API_KEY"] = api_key

    if not model:
        raise ValueError("--model is required (or set in --api_config). For Bedrock: bedrock/converse/<ARN>")

    litellm.drop_params = True
    litellm.modify_params = True
    litellm.num_retries = args.litellm_retries

    # Resolve the best cost lookup key for litellm.completion_cost()
    cost_base_model = ""
    if base_model:
        region = model.split(":")[3] if model.count(":") >= 4 else ""
        for key in [f"bedrock/{region}/{base_model}" if region else None, f"bedrock/{base_model}", base_model]:
            if key and key in litellm.model_cost:
                cost_base_model = key
                break
        if not cost_base_model:
            cost_base_model = f"bedrock/{base_model}"
        print(f"  [COST] Using base_model='{cost_base_model}' for pricing")

        litellm.register_model({
            model: {
                "litellm_provider": "bedrock",
                "mode": "chat",
                "base_model": cost_base_model,
            }
        })

    tasks: List[Path] = []
    if args.task_json:
        tasks = [Path(args.task_json)]
    elif args.task_dir:
        tasks = sorted(Path(args.task_dir).glob("*.json"))
    else:
        raise ValueError("Provide --task_json or --task_dir")

    dataset_root = Path(args.dataset_root) if args.dataset_root else None
    images_dir = Path(args.images_dir) if args.images_dir else None
    
    # Auto-generate out_dir based on mode and model if not specified
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        # Use short model name for directory (handles long ARNs)
        model_name = _short_model_name(model)
        out_dir = Path(f"runs/general/{model_name}")

    # Apply sharding if specified (contiguous blocks, not interleaved)
    if args.num_shards > 1:
        total = len(tasks)
        shard_size = (total + args.num_shards - 1) // args.num_shards  # ceiling division
        start_idx = args.shard * shard_size
        end_idx = min(start_idx + shard_size, total)
        tasks = tasks[start_idx:end_idx]
        print(f"[Shard {args.shard}/{args.num_shards}] Tasks {start_idx}-{end_idx-1} ({len(tasks)} tasks)")

    results = []
    skipped = 0
    processed = 0
    rate_limiter = TaskRateLimiter(args.tasks_per_minute)
    for idx, t in enumerate(tasks):
        if args.max_tasks > 0 and processed >= args.max_tasks:
            print(f"[LIMIT] Reached max_tasks={args.max_tasks}, stopping")
            break
        
        if args.skip_existing:
            run_dir = out_dir / t.stem
            if (run_dir / "run_meta.json").exists():
                skipped += 1
                print(f"[SKIP] {t.name} (already completed)")
                continue
        
        rate_limiter.wait_if_needed()
        
        try:
            ds_root = resolve_dataset_root(t, dataset_root)
            
            for attempt in range(args.max_retries):
                try:
                    res = run_one_rollout(
                        task_json=t,
                        dataset_root=ds_root,
                        images_dir=images_dir,
                        out_dir=out_dir,
                        model=model,
                        temperature=args.temperature,
                        python_exe=args.python,
                        enable_search=not args.no_search,
                        search_cfg_path=Path(args.search_config) if args.search_config else None,
                        max_rounds=args.max_rounds,
                        max_tool_calls=args.max_tool_calls,
                        max_image_pixels=args.max_image_pixels,
                        image_quality=args.image_quality,
                        use_native_tools=args.use_native_tools,
                        api_key=api_key,
                        cost_base_model=cost_base_model,
                        api_timeout=args.timeout,
                    )
                    results.append(res)
                    processed += 1
                    rate_limiter.record_task()
                    cost_str = f" | ${res.get('usage', {}).get('total_cost_usd', 0):.6f}" if res.get("usage", {}).get("total_cost_usd") else ""
                    print(f"[OK] {t.name} -> {Path(res['paths']['run_dir']).name}{cost_str}")
                    break
                    
                except Exception as e:
                    error_str = str(e)
                    if "429" in error_str or "RateLimitError" in str(type(e).__name__):
                        if attempt < args.max_retries - 1:
                            wait_time = 5 * (2 ** attempt) * (1 + random.uniform(0, 0.25))
                            print(f"[RATE LIMIT] {t.name}: Waiting {wait_time:.1f}s before retry {attempt + 1}/{args.max_retries}...")
                            time.sleep(wait_time)
                        else:
                            print(f"[ERR] {t}: Rate limit exceeded after {args.max_retries} retries")
                            raise
                    else:
                        raise
                        
            if idx < len(tasks) - 1 and args.task_delay > 0:
                time.sleep(args.task_delay)
                
        except Exception as e:
            print(f"[ERR] {t}: {e}")
            traceback.print_exc()

    ensure_dir(out_dir)
    
    # Print summary
    print(f"\n=== Summary ===")
    print(f"Completed: {len(results)}")
    print(f"Skipped: {skipped}")
    print(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
