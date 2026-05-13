#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# LiteLLM-based atomic mode runner for Agentic-MME
# Supports any LiteLLM-compatible model including AWS Bedrock ARNs
# Usage: python atomic/run_atomic_tools_litellm.py --model bedrock/converse/<ARN> --task_dir ...

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
import traceback
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import litellm

# Add parent directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from PIL import Image

from common_utils import ensure_dir, read_json, write_json, image_to_data_url, utc_ts, get_adaptive_image_params
from dataset_utils import resolve_dataset_root, resolve_image_path
from atomic_toolbox import (
    AtomicState, build_atomic_tools_schema,
    tool_crop, tool_rotate, tool_flip, tool_resize, tool_enhance,
    tool_grayscale, tool_autocontrast, tool_blur, tool_sharpen,
    tool_denoise, tool_edge_detect, tool_invert, tool_equalize, tool_threshold
)

# Search tools
from search_toolbox import build_search_tools_schema
from search_tools import SearchTools, load_search_config


def _region_from_arn(arn: str) -> str:
    parts = arn.split(":")
    if len(parts) >= 4 and parts[3]:
        return parts[3]
    return "us-east-1"


# All image tool names
IMAGE_TOOLS = {
    "crop", "rotate", "flip", "resize", "enhance",
    "grayscale", "autocontrast", "blur", "sharpen",
    "denoise", "edge_detect", "invert", "equalize", "threshold"
}

# Maximum number of images that can be downloaded per task
MAX_DOWNLOAD_IMAGES = 5

SYSTEM_PROMPT = """You are a multimodal reasoning agent with access to image manipulation and web search tools.

## Image Management
- Images are tracked by index: Image 0 is the original input, Images 1, 2, ... are processed results
- Image N corresponds to transformed_image_N.png (e.g., Image 1 = transformed_image_1.png)
- Each tool operation creates a NEW image with a new index
- You must specify which image to operate on using `image_index` parameter
- After each operation, you'll see the new image and its index

## Image Tools (function calling)
All tools require `image_index` to specify which image to operate on.

Geometric transformations:
- crop(image_index, bbox_2d, zoom_scale?, label?) - Crop a region using normalized coordinates [x1,y1,x2,y2] in 0-1000 scale
- rotate(image_index, angle, expand?, label?) - Rotate the image
- flip(image_index, direction?, label?) - Flip/mirror the image (horizontal/vertical/both)
- resize(image_index, width?, height?, scale?, label?) - Resize the image

Enhancement/filtering:
- enhance(image_index, brightness?, contrast?, sharpness?, label?) - Adjust brightness/contrast/sharpness (1.0=no change)
- grayscale(image_index, label?) - Convert to grayscale
- autocontrast(image_index, cutoff?, label?) - Automatic contrast adjustment
- blur(image_index, radius?, label?) - Apply Gaussian blur
- sharpen(image_index, label?) - Apply sharpening filter
- denoise(image_index, strength?, label?) - Remove noise
- edge_detect(image_index, method?, label?) - Detect edges (canny/sobel/simple)
- invert(image_index, label?) - Invert colors (negative)
- equalize(image_index, label?) - Equalize histogram
- threshold(image_index, value?, mode?, label?) - Convert to binary

## Coordinate System
- bbox_2d uses normalized coordinates: [x1, y1, x2, y2] where each value is 0-1000
- (0, 0) is top-left, (1000, 1000) is bottom-right
- Example: [250, 250, 750, 750] crops the center 50% of the image

## Web Search Tools
- google_search(query, gl?, hl?) - Text-based web search
- google_lens_search(image_index?) - Reverse image search on specified image (default: 0 = original)
- fetch_webpage(url, max_chars?) - Fetch webpage content
# - download_image(url) - Download an image from URL (max 5 per task). [DISABLED]

## Workflow
1. Analyze the image (Image 0) and question
2. Use tools as needed, always specifying image_index
3. After each tool, you'll see the result and new image index
4. Continue until you have enough information
5. Provide your final answer with this REQUIRED format: <answer>YOUR_FINAL_ANSWER</answer>
"""

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
_ANSWER_UNCLOSED_RE = re.compile(r"<answer>(.*)", re.IGNORECASE | re.DOTALL)


def _extract_answer_text(text: str) -> str:
    """Extract final answer text with XML-tag preference and safe fallbacks."""
    raw = text or ""
    m = _ANSWER_RE.search(raw)
    if m:
        return (m.group(1) or "").strip()

    m2 = _ANSWER_UNCLOSED_RE.search(raw)
    if m2:
        return (m2.group(1) or "").strip()

    # For error analysis, preserve raw last-turn output even without <answer>.
    return raw.strip()


def _short_model_name(model: str) -> str:
    """Create a short directory-friendly model name from potentially long ARN."""
    if "inference-profile/" in model:
        return model.split("inference-profile/")[-1][:40]
    if "provisioned-model/" in model:
        return model.split("provisioned-model/")[-1][:40]
    name = model.replace("bedrock/", "")
    return name.replace("/", "_").replace(":", "_")[:60]


def dispatch_image_tool(state: AtomicState, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Dispatch image tool call to the appropriate function."""
    image_index = int(args.get("image_index", 0))
    label = str(args.get("label", ""))
    
    if name == "crop":
        bbox = args.get("bbox_2d", [0, 0, 1000, 1000])
        zoom_scale = float(args.get("zoom_scale", 1.0))
        return tool_crop(state, image_index, bbox, label, zoom_scale)
    if name == "rotate":
        return tool_rotate(state, image_index, float(args["angle"]), bool(args.get("expand", True)), label)
    if name == "flip":
        return tool_flip(state, image_index, str(args.get("direction", "horizontal")), label)
    if name == "resize":
        return tool_resize(state, image_index, args.get("width"), args.get("height"), args.get("scale"), label)
    if name == "enhance":
        return tool_enhance(state, image_index, args.get("brightness"), args.get("contrast"), args.get("sharpness"), label)
    if name == "grayscale":
        return tool_grayscale(state, image_index, label)
    if name == "autocontrast":
        return tool_autocontrast(state, image_index, float(args.get("cutoff", 0)), label)
    if name == "blur":
        return tool_blur(state, image_index, int(args.get("radius", 2)), label)
    if name == "sharpen":
        return tool_sharpen(state, image_index, label)
    if name == "denoise":
        return tool_denoise(state, image_index, int(args.get("strength", 10)), label)
    if name == "edge_detect":
        return tool_edge_detect(state, image_index, str(args.get("method", "canny")), label)
    if name == "invert":
        return tool_invert(state, image_index, label)
    if name == "equalize":
        return tool_equalize(state, image_index, label)
    if name == "threshold":
        return tool_threshold(state, image_index, int(args.get("value", 128)), str(args.get("mode", "binary")), label)
    raise ValueError(f"Unknown image tool: {name}")


def _resolve_image_for_lens(state: AtomicState, args: Dict[str, Any]) -> str:
    """Resolve image path for google_lens_search."""
    # Support image_index parameter (0-based)
    if "image_index" in args:
        idx = int(args["image_index"])
        return str(state.get_image(idx))
    
    # Legacy support for image_ref
    image_ref = (args.get("image_ref") or "").strip().lower()
    if image_ref in {"current", "cur", "latest"}:
        # Return the last image
        return str(state.images[-1][0]) if state.images else str(state.orig_path)
    if image_ref in {"orig", "original", "input"}:
        return str(state.orig_path)
    
    # Default to original (index 0)
    return str(state.orig_path)


class TaskRateLimiter:
    """Sliding-window rate limiter for task execution.

    Tracks timestamps of completed tasks and blocks if the window is full.
    Uses time.monotonic() for clock stability.

    Usage:
        rl = TaskRateLimiter(tasks_per_minute=2)
        for task in tasks:
            rl.wait_if_needed()   # blocks if window full
            run_task(task)
            rl.record_task()      # record completion
    """

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
            wait = 60.0 - elapsed + 0.1  # small buffer
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


def _recompress_messages(messages: List[Dict[str, Any]], level: int, state: "AtomicState") -> None:
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
                if img_idx < len(state.images):
                    src_path = Path(state.images[img_idx][0])
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
    state: "AtomicState",
    timeout: int = 300,
    max_timeout_retries: int = 3,
) -> Any:
    """Call litellm.completion with payload-size retry and timeout retry.
    
    Retries on:
      - Payload too large: recompresses images at degraded quality levels
      - Timeout/connection errors: retries up to max_timeout_retries with backoff
    """
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
                _recompress_messages(messages, next_level, state)
                break  # Break inner timeout loop, continue outer level loop
    # Final attempt after all degrade levels exhausted
    api_kwargs["timeout"] = timeout
    return litellm.completion(**api_kwargs)



def run_one(
    task_json: Path,
    dataset_root: Path,
    images_dir: Optional[Path],
    out_dir: Path,
    model: str,
    temperature: float,
    max_rounds: int,
    max_tool_calls: int,
    enable_search: bool,
    search_cfg_path: Optional[Path],
    api_key: str = "",
    max_image_pixels: int = 2048 * 2048,
    image_quality: int = 95,
    cost_base_model: str = "",
    api_timeout: int = 300,
) -> Dict[str, Any]:
    """Run atomic tools mode for one task with multi-turn conversation."""
    
    task_cfg = read_json(task_json)
    img_path_result = resolve_image_path(task_json, task_cfg, dataset_root, images_dir)
    
    # Handle single or multiple images
    if isinstance(img_path_result, list):
        img_paths = img_path_result
    else:
        img_paths = [img_path_result]

    run_dir = ensure_dir(out_dir / task_json.stem)
    tool_images_dir = ensure_dir(run_dir / "tool_images")

    # Copy original image(s)
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

    # Initialize atomic state with image tracking
    # For multi-image cases, we use the first image as the primary original
    # Additional images are added to state as well
    state = AtomicState(orig_path=orig_copy, processed_dir=tool_images_dir)
    
    # Add additional original images to state (for multi-image tasks)
    for i, oc in enumerate(orig_copies[1:], start=1):
        # Add as additional original images with index i
        state.images.append((oc, f"original input image {i+1}"))

    # Build tools list
    tools = build_atomic_tools_schema()
    
    # Update search tools schema to use image_index
    search_tools_schema = [
        {
            "type": "function",
            "function": {
                "name": "google_search",
                "description": "Search the web using Google. Use for facts, current information, specifications, prices.",
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
                "description": "Reverse image search using Google Lens. Use to identify objects, brands, logos, landmarks, products.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image_index": {
                            "type": "integer",
                            "description": "Index of the image to search (0 = original, 1, 2... = processed images)",
                            "minimum": 0
                        },
                    },
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "fetch_webpage",
                "description": "Fetch and read the content of a webpage.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "The webpage URL to fetch."},
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
        #         "description": "Download an image from URL. Max 5 images per task. [DISABLED]",
        #         "parameters": {
        #             "type": "object",
        #             "properties": {
        #                 "url": {"type": "string", "description": "The image URL to download."},
        #             },
        #             "required": ["url"],
        #         },
        #     },
        # },
    ]
    
    search_tools: Optional[SearchTools] = None
    if enable_search:
        tools = tools + search_tools_schema
        cfg = load_search_config(str(search_cfg_path) if search_cfg_path else None)
        if not cfg.cache_dir:
            cfg.cache_dir = str(ensure_dir(run_dir / "search_cache"))
        # Pass task_id for organized cache naming: _search_cache/{task_id}/serper_search_1.json
        task_id = task_cfg.get("task_id", "") or run_dir.name
        search_tools = SearchTools(cfg, task_id=task_id)

    prompt = (task_cfg.get("input") or {}).get("prompt", "")

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

    # Initialize messages with image index info
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    # Track conversation history for JSON export (without base64)
    conversation_history: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": history_text},
    ]

    tool_use_list: List[Dict[str, Any]] = []
    usage = {"api_calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "total_cost_usd": 0.0}
    tool_calls_total = 0
    download_count = 0  # Track number of downloaded images
    final_answer = ""
    all_warnings: List[str] = []
    tool_budget_exhausted = False  # Disable tools once call budget is consumed
    serper_credit_warned = False  # Only warn once for Serper credit exhaustion
    force_final_answer_mode = False  # When true, disable tools and force answer generation

    def _request_final_answer(reason: str, disable_tools: bool = False) -> None:
        nonlocal force_final_answer_mode
        if disable_tools:
            force_final_answer_mode = True
        prompt_text = (
            f"{reason} Please provide your final answer now with NO additional tool calls. "
            f"Use EXACT format: <answer>YOUR_FINAL_ANSWER</answer>."
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
                f"Serper API credits exhausted (turn {turn_idx}, tool={tool_name}). "
                f"Search quality is degraded; top up Serper credits."
            )

    for turn in range(max_rounds):
        api_kwargs = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 4000,
            "api_key": api_key or None,
            "aws_region_name": _region_from_arn(model),
        }
        if not tool_budget_exhausted and not force_final_answer_mode:
            api_kwargs["tools"] = tools
            api_kwargs["tool_choice"] = "auto"

        resp = _completion_with_payload_retry(api_kwargs, messages, state, timeout=api_timeout, max_timeout_retries=10)
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

        msg = resp.choices[0].message
        raw_content = msg.content or ""
        tool_calls_from_api = getattr(msg, "tool_calls", None) or []

        # Save raw output
        raw_parts = [raw_content] if raw_content else []
        if tool_calls_from_api:
            raw_parts.append("\n--- Function Calls ---")
            for tc in tool_calls_from_api:
                raw_parts.append(f"Tool: {tc.function.name}")
                raw_parts.append(f"Arguments: {tc.function.arguments}")
        (run_dir / f"raw_model_output_turn_{turn}.txt").write_text("\n".join(raw_parts), encoding="utf-8")

        # Process tool calls
        if tool_calls_from_api:
            messages.append(msg.model_dump())
            if raw_content:
                conversation_history.append({"role": "assistant", "content": raw_content})

            hit_tool_budget_this_turn = False
            pending_image_parts = []  # Collect image feedback to send as ONE user message after all tools
            for tc in tool_calls_from_api:
                tool_calls_total += 1
                if tool_calls_total > max_tool_calls:
                    all_warnings.append(f"Exceeded max_tool_calls={max_tool_calls}")
                    tool_budget_exhausted = True
                    hit_tool_budget_this_turn = True
                    break

                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except:
                    args = {}

                # Track tool call
                conversation_history.append({
                    "role": "tool_call",
                    "content": json.dumps({"name": name, "arguments": args}, ensure_ascii=False),
                })

                # Execute tool
                if name in IMAGE_TOOLS:
                    # Image tool
                    try:
                        out = dispatch_image_tool(state, name, args)
                        new_index = out.get("new_image_index", len(state.images))
                        out_path = Path(out["output_path"])
                        
                        tool_use_list.append({
                            "index": len(tool_use_list),
                            "tool_name": name,
                            "raw_tool_name": name,
                            "arguments": args,
                            "output": out,
                            "turn": turn,
                            "timestamp": utc_ts(),
                        })

                        # Send tool result
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": json.dumps(out, ensure_ascii=False),
                        })

                        # Queue image feedback for batched user message
                        label = out.get("label", "")
                        img_info = f"[Image {new_index}: {name}"
                        if label:
                            img_info += f" - {label}"
                        img_info += "]"
                        
                        img_count = sum(1 for m in messages if isinstance(m.get("content"), list) 
                                       and any(c.get("type") == "image_url" for c in m["content"] if isinstance(c, dict)))
                        adaptive_params = get_adaptive_image_params(img_count)
                        
                        pending_image_parts.append(
                            {"type": "image_url", "image_url": {"url": image_to_data_url(
                                out_path, 
                                max_pixels=adaptive_params["max_pixels"],
                                quality=adaptive_params["quality"],
                                max_size_mb=adaptive_params["max_size_mb"]
                            )}}
                        )
                        pending_image_parts.append({"type": "text", "text": img_info})

                        # Track in history
                        conversation_history.append({
                            "role": "tool_response",
                            "content": json.dumps(out, ensure_ascii=False),
                        })
                        conversation_history.append({
                            "role": "user",
                            "content": f"{img_info} - {out_path.name}",
                        })
                        
                    except Exception as e:
                        error_out = {"ok": "false", "error": str(e)}
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": json.dumps(error_out, ensure_ascii=False),
                        })
                        conversation_history.append({
                            "role": "tool_response",
                            "content": json.dumps(error_out, ensure_ascii=False),
                        })
                        all_warnings.append(f"Tool {name} error: {e}")

                else:
                    # Search tool - wrap in try/except to handle network errors gracefully
                    try:
                        if not enable_search or search_tools is None:
                            out = {"ok": False, "error": "Search is not enabled"}
                        elif name == "google_search":
                            out = search_tools.google_search(
                                query=str(args.get("query", "")),
                                gl=args.get("gl"),
                                hl=args.get("hl"),
                            )
                        elif name == "google_lens_search":
                            img_path_for_lens = _resolve_image_for_lens(state, args)
                            out = search_tools.google_lens_search(image_path=img_path_for_lens)
                        elif name == "fetch_webpage":
                            out = search_tools.fetch_webpage(
                                url=str(args.get("url", "")),
                                max_chars=int(args.get("max_chars", 12000) or 12000),
                            )
                        # elif name == "download_image":
                        #     # Check download limit (DISABLED)
                        #     if download_count >= MAX_DOWNLOAD_IMAGES:
                        #         out = {"ok": False, "error": f"Download limit reached. Maximum {MAX_DOWNLOAD_IMAGES} images per task."}
                        #     else:
                        #         url = str(args.get("url", ""))
                        #         if not url:
                        #             out = {"ok": False, "error": "download_image requires 'url' argument"}
                        #         else:
                        #             from search_tools import download_image_from_url
                        #             out = download_image_from_url(
                        #                 url=url,
                        #                 save_dir=str(tool_images_dir),
                        #                 timeout_s=30,
                        #             )
                        #             if isinstance(out, dict) and out.get("ok"):
                        #                 download_count += 1
                        #                 # Add downloaded image to state
                        #                 if "path" in out:
                        #                     downloaded_path = Path(out["path"])
                        #                     if downloaded_path.exists():
                        #                         new_idx = len(state.images)
                        #                         state.images.append((downloaded_path, f"downloaded from {url[:50]}"))
                        else:
                            out = {"ok": False, "error": f"Unknown tool: {name}"}
                    except Exception as search_err:
                        # Handle network errors (timeout, SSL, etc.) gracefully
                        error_msg = str(search_err)
                        lower_msg = error_msg.lower()
                        if "not enough credits" in lower_msg:
                            out = {
                                "ok": False,
                                "error": (
                                    "Serper API credits exhausted (Not enough credits). "
                                    "Please top up Serper credits or disable search for this run."
                                ),
                            }
                        elif "504" in error_msg or "timeout" in lower_msg:
                            out = {"ok": False, "error": f"Search service timeout. Please try again or use a different approach."}
                        elif "SSL" in error_msg or "ssl" in lower_msg:
                            out = {"ok": False, "error": f"Network error accessing the URL. The website may be unavailable."}
                        else:
                            out = {"ok": False, "error": f"Search failed: {error_msg[:200]}"}
                        all_warnings.append(f"Search tool {name} error: {error_msg}")
                    _maybe_warn_serper_credit(name, out, turn)

                    tool_use_list.append({
                        "index": len(tool_use_list),
                        "tool_name": name,
                        "raw_tool_name": name,
                        "arguments": args,
                        "output": out,
                        "turn": turn,
                        "timestamp": utc_ts(),
                    })

                    # Extract compact content for model
                    if isinstance(out, dict) and "context" in out:
                        model_content = out["context"]
                    elif isinstance(out, dict) and "text" in out:
                        model_content = out["text"]
                    elif isinstance(out, dict) and "error" in out:
                        model_content = f"Error: {out['error']}"
                    else:
                        model_content = json.dumps(out, ensure_ascii=False)

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": model_content,
                    })

                    conversation_history.append({
                        "role": "tool_response",
                        "content": model_content,
                    })

            if hit_tool_budget_this_turn:
                if pending_image_parts:
                    messages.append({"role": "user", "content": pending_image_parts})
                if turn < max_rounds - 1:
                    _request_final_answer(
                        f"Tool-call budget has been exhausted ({max_tool_calls}).",
                        disable_tools=True,
                    )
                    continue
                break

            # Send all queued image feedback as a single user message
            if pending_image_parts:
                messages.append({"role": "user", "content": pending_image_parts})

            # Proactive convergence: after the penultimate reasoning round, force final answer mode.
            if turn >= max_rounds - 2 and turn < max_rounds - 1:
                _request_final_answer(
                    "Only one model round remains.",
                    disable_tools=True,
                )
                continue

            continue  # Next turn

        # No tool calls - this is the final answer
        messages.append({"role": "assistant", "content": raw_content})
        conversation_history.append({"role": "assistant", "content": raw_content})

        parsed_answer = _extract_answer_text(raw_content)
        if parsed_answer:
            final_answer = parsed_answer
            break

        all_warnings.append(f"[turn {turn}] Empty assistant response with no tools")
        if turn < max_rounds - 1:
            _request_final_answer("Your previous response was empty.", disable_tools=True)
            continue
        break

    # If still no answer (including early breaks), request final answer
    if not final_answer:
        final_prompt = (
            "Please provide your final answer now based on all the information gathered. "
            "Use EXACT format: <answer>YOUR_FINAL_ANSWER</answer>."
        )
        messages.append({"role": "user", "content": final_prompt})
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
            final_resp = _completion_with_payload_retry(final_kwargs, messages, state, timeout=api_timeout, max_timeout_retries=10)
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
            (run_dir / "raw_model_output_final.txt").write_text(final_raw, encoding="utf-8")
            conversation_history.append({"role": "assistant", "content": final_raw})
            final_answer = _extract_answer_text(final_raw)
        except Exception as e:
            all_warnings.append(f"Failed to get final answer: {e}")

    # Save outputs
    (run_dir / "model_answer.txt").write_text(final_answer, encoding="utf-8")
    write_json(run_dir / "tool_use_list.json", tool_use_list)

    # Save image index tracking
    image_list = state.get_image_list()
    write_json(run_dir / "image_index.json", image_list)

    # Save conversation history as formatted JSON
    tools_json = json.dumps(tools, ensure_ascii=False)
    conversation_record = {
        "task_id": task_cfg.get("task_id", ""),
        "tools": tools_json,
        "images": image_list,
        "messages": conversation_history,
        "final_answer": final_answer,
    }
    with open(run_dir / "conversation.json", "w", encoding="utf-8") as f:
        json.dump(conversation_record, f, ensure_ascii=False, indent=2)

    # Build tool statistics
    image_tool_hist: Dict[str, int] = {}
    search_tool_hist: Dict[str, int] = {}
    for ev in tool_use_list:
        tname = ev.get("tool_name", "")
        if tname in IMAGE_TOOLS:
            image_tool_hist[tname] = image_tool_hist.get(tname, 0) + 1
        elif tname in ("google_search", "google_lens_search", "fetch_webpage"):  # , "download_image"):
            search_tool_hist[tname] = search_tool_hist.get(tname, 0) + 1

    run_meta = {
        "task_id": task_cfg.get("task_id", ""),
        "task_file": str(task_json.resolve()),
        "mode": "atomic",
        "driver": "tools",
        "model": model,
        "temperature": temperature,
        "usage": usage,
        "effective_tool_calls": len(tool_use_list),
        "total_images": len(state.images),
        "paths": {
            "run_dir": str(run_dir),
            "orig": str(orig_copy),
            "processed_dir": str(tool_images_dir),
        },
        "image_tool_analysis": {
            "image_tool_calls": sum(image_tool_hist.values()),
            "image_tool_hist": image_tool_hist,
        },
        "search_analysis": {
            "search_enabled": bool(enable_search),
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
    ap.add_argument("--tasks", nargs="+", default=[],
                    help="Specific task IDs to run (stems without .json, e.g. 0001 0005 0012)")
    ap.add_argument("--dataset_root", type=str, default="")
    ap.add_argument("--images_dir", type=str, default="")
    ap.add_argument("--out_dir", type=str, default="", help="Output directory. Default: runs/atomic/{model_name}")
    ap.add_argument("--model", type=str, default="")
    ap.add_argument("--temperature", type=float, default=0.0)
    
    # Multi-turn settings
    ap.add_argument("--max_rounds", type=int, default=50, help="Max model turns")
    ap.add_argument("--max_tool_calls", type=int, default=50, help="Max total tool calls")
    
    # Rate limiting
    ap.add_argument("--task_delay", type=float, default=2.0, help="Delay between tasks (seconds)")
    ap.add_argument("--max_retries", type=int, default=3, help="Max retries for rate limit errors")
    ap.add_argument("--tasks_per_minute", type=int, default=0, help="Max tasks per 60s sliding window (0=unlimited)")
    ap.add_argument("--litellm_retries", type=int, default=3, help="LiteLLM SDK internal retry count")
    ap.add_argument("--timeout", type=int, default=300, help="API call timeout in seconds (default 300s/5min)")

    # LiteLLM API key (used for Bedrock and other providers)
    ap.add_argument("--api_key", type=str, default="", help="API key for the model provider")
    ap.add_argument("--api_config", type=str, default="", help="Path to JSON config with {api_key, model, base_model} (e.g. configs/api.json)")

    # Search options
    ap.add_argument("--enable_search", action="store_true", default=True, help="Enable web search tools (default: enabled)")
    ap.add_argument("--no_search", action="store_true", default=False, help="Disable web search tools")
    ap.add_argument("--search_config", type=str, default="configs/search_config.json", help="Path to search config JSON")

    # Image settings
    ap.add_argument("--max_image_pixels", type=int, default=2048*2048)
    ap.add_argument("--image_quality", type=int, default=95)
    
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

    if args.tasks:
        task_stems = set(args.tasks)
        tasks = [t for t in tasks if t.stem in task_stems]

    dataset_root = Path(args.dataset_root) if args.dataset_root else None
    images_dir = Path(args.images_dir) if args.images_dir else None
    
    # Auto-generate out_dir based on mode and model if not specified
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        # Use short model name for directory (handles long ARNs)
        model_name = _short_model_name(model)
        out_dir = Path(f"runs/atomic/{model_name}")
    
    search_cfg_path = Path(args.search_config) if args.search_config else None
    
    # Apply sharding if specified (contiguous blocks, not interleaved)
    if args.num_shards > 1:
        total = len(tasks)
        shard_size = (total + args.num_shards - 1) // args.num_shards  # ceiling division
        start_idx = args.shard * shard_size
        end_idx = min(start_idx + shard_size, total)
        tasks = tasks[start_idx:end_idx]
        print(f"[Shard {args.shard}/{args.num_shards}] Tasks {start_idx}-{end_idx-1} ({len(tasks)} tasks)")

    all_meta = []
    skipped = 0
    processed = 0
    rate_limiter = TaskRateLimiter(args.tasks_per_minute)
    for idx, t in enumerate(tasks):
        # Check max_tasks limit
        if args.max_tasks > 0 and processed >= args.max_tasks:
            print(f"[LIMIT] Reached max_tasks={args.max_tasks}, stopping")
            break
        
        # Skip if already completed (both run_meta.json and model_answer.txt must exist)
        if args.skip_existing:
            run_dir = out_dir / t.stem
            has_meta = (run_dir / "run_meta.json").exists()
            has_answer = (run_dir / "model_answer.txt").exists()
            if has_meta and has_answer:
                skipped += 1
                print(f"[SKIP] {t.name} (already completed)")
                continue
            elif has_meta or has_answer or run_dir.exists():
                print(f"[RETRY] {t.name} (incomplete run, retrying)")
        
        rate_limiter.wait_if_needed()
        
        try:
            ds = resolve_dataset_root(t, dataset_root)
            
            for attempt in range(args.max_retries):
                try:
                    m = run_one(
                        task_json=t,
                        dataset_root=ds,
                        images_dir=images_dir,
                        out_dir=out_dir,
                        model=model,
                        temperature=args.temperature,
                        max_rounds=args.max_rounds,
                        max_tool_calls=args.max_tool_calls,
                        enable_search=not args.no_search,
                        search_cfg_path=search_cfg_path,
                        api_key=api_key,
                        max_image_pixels=args.max_image_pixels,
                        image_quality=args.image_quality,
                        cost_base_model=cost_base_model,
                        api_timeout=args.timeout,
                    )
                    all_meta.append(m)
                    processed += 1
                    rate_limiter.record_task()
                    cost_str = f" | ${m.get('usage', {}).get('total_cost_usd', 0):.6f}" if m.get("usage", {}).get("total_cost_usd") else ""
                    print(f"[OK] {t.name} -> {out_dir / t.stem}{cost_str}")
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
    print(f"Completed: {len(all_meta)}")
    print(f"Skipped: {skipped}")
    print(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
