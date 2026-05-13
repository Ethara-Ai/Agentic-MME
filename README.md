---
license: cc-by-nc-nd-4.0
language:
- en
pretty_name: Janus
task_categories:
- visual-question-answering
- image-to-text
tags:
- multimodal
- agentic
- reinforcement-learning
- process-reward
- tool-use
- image-manipulation
- visual-reasoning
- benchmark
size_categories:
- n<1K
---
---

<div align="center">

# Janus

### Process-Level RL Environments for Multimodal Agentic Intelligence

<p>
  <a href="https://arxiv.org/abs/2604.03016"><img src="https://img.shields.io/badge/Paper-arXiv%3A2604.03016-b31b1b?logo=arxiv&logoColor=white" alt="Paper"/></a>
  <a href="#license"><img src="https://img.shields.io/badge/License-CC--BY--NC--ND%204.0-lightgrey.svg" alt="License"/></a>
  <img src="https://img.shields.io/badge/Instances-20-blue" alt="Instances"/>
  <img src="https://img.shields.io/badge/Domains-14-orange" alt="Domains"/>
  <img src="https://img.shields.io/badge/Tools-17-green" alt="Tools"/>
</p>

*Training multimodal agents to actively manipulate images and coordinate open-web retrieval, with human-curated stepwise reward signals grounded in real-world tasks.*

</div>

---

## Overview

**Janus** is a collection of reinforcement learning environment built on the Agentic-MME framework. Each environment presents a real-world multimodal task requiring active visual manipulation (cropping, rotating, enhancing) coordinated with open-web knowledge retrieval.

Unlike static question-answering setups, Janus provides **process-level reward signals** at every intermediate step. Agents receive granular feedback on tool selection, visual artifact quality, and retrieval effectiveness, enabling RL training that rewards *how* a problem is solved, not just the final answer.

> **Note.** Janus is an independent RL environment suite built on the Agentic-MME methodology (Wei et al., 2026). The task schema, tool interface, and process-verification protocol are inherited from the upstream work; see Citation for proper attribution.

---

## Motivation

Current multimodal training relies heavily on outcome-level supervision: the agent either gets the right answer or it does not. This coarse signal fails to teach:

- **Active perception**: knowing *when* to crop, rotate, or enhance an image to surface hidden evidence
- **Tool coordination**: chaining visual operations with web retrieval in the correct order
- **Efficient execution**: solving problems without redundant tool calls or overthinking loops
- **Cross-modal reasoning**: using visual cues to drive retrieval queries, and retrieved knowledge to guide further visual inspection

Janus addresses these gaps by providing dense, human-curated reward signals at every interaction step, enabling RL methods to train agents that learn structured problem-solving workflows rather than outcome-guessing.

---

## Environment Statistics

| Property                  | Value                                  |
| ------------------------- | -------------------------------------- |
| Total Environments        | 20                                     |
| Source Domains            | 14                                     |
| Difficulty Levels         | L1 (8), L2 (8), L3 (4)                 |
| Turn Type                 | Single-turn                            |
| Visual Tools Available    | 13                                     |
| Retrieval Tools Available | 4                                      |
| Primary Signal            | Process-level reward (S-axis + V-axis) |

### Domain Breakdown

| Domain              | Environments | Task Pattern                          |
| ------------------- | ------------ | ------------------------------------- |
| Document            | 3            | Mixed (L1 visual-only + L2 search)    |
| Automotive          | 3            | Mixed (L1 visual-only + L3 iterative) |
| Electronics         | 2            | Visual + Retrieval                    |
| Retail              | 2            | Mixed (L1 visual-only + L2 search)    |
| Aviation            | 1            | Visual + Retrieval                    |
| Telecom             | 1            | Iterative (L3)                        |
| Technology          | 1            | Iterative (L3)                        |
| Numismatics         | 1            | Visual + Retrieval                    |
| Entertainment       | 1            | Visual + Retrieval                    |
| Sports              | 1            | Iterative (L3)                        |
| Puzzle              | 1            | Visual-only                           |
| General             | 1            | Visual-only                           |
| Urban               | 1            | Visual-only                           |
| Visual Puzzle & OCR | 1            | Visual-only                           |

### Task Patterns

| Level                | Count | Description                                                                                          |
| -------------------- | ----- | ---------------------------------------------------------------------------------------------------- |
| L1 - Visual-only     | 8     | Agent solves the task entirely through image manipulation (crop, rotate, enhance) and direct reading |
| L2 - Visual + Search | 8     | Agent must manipulate images to extract evidence, then search the web to verify or expand            |
| L3 - Iterative       | 4     | Agent performs interleaved visual manipulation and web search in hypothesis-verification loops       |

---

## Reward Signal Design

Each environment provides two orthogonal reward axes, human-curated and verified:

### S-axis (Strategy and Retrieval)

For every interaction step, the S-axis checks:

- Was the correct tool selected?
- Were the tool arguments faithful to the task intent?
- For retrieval steps: did the query retrieve the expected intermediate knowledge?

### V-axis (Visual Evidence)

For every visual operation, the V-axis checks:

- **code_check**: AST verification that the agent executed the required visual operation (crop, enhance, rotate, flip)
- **visual_check**: VLM judge confirms the intermediate artifact contains the target evidence
- **answer_check**: Exact, contains, or numeric match against the expected answer

### Overthink Penalty

Measures efficiency relative to the human-curated minimal solution:

```
penalty = (agent_tool_calls - reference_tool_calls) * penalty_per_extra_call
```

Where reference_tool_calls is the minimal human-curated count (typically 2-3) and max_tool_calls caps the episode (typically 5-7).

---

### Instance Schema

Each JSON entry contains the following structure:

#### Core Task Fields

| Field                        | Type   | Description                                                                          |
| ---------------------------- | ------ | ------------------------------------------------------------------------------------ |
| `task_id`                  | string | Unique identifier (e.g.,`L2_Aviation_AircraftRegistration_NamesakeBirthYear_0044`) |
| `meta.level`               | int    | Difficulty level (1, 2, or 3)                                                        |
| `meta.domain`              | string | Source domain (e.g., Numismatics, Electronics, Retail)                               |
| `meta.turncase`            | string | Interaction type (`single_turn`)                                                   |
| `meta.original_image_link` | string | Source image link for provenance                                                     |
| `input.prompt`             | string | Natural-language task description                                                    |
| `input.image`              | string | Relative path to input image                                                         |
| `golden_answer.value`      | string | Ground-truth final answer                                                            |

#### Process Reward Fields

| Field                                             | Type         | Description                                                                             |
| ------------------------------------------------- | ------------ | --------------------------------------------------------------------------------------- |
| `process_evaluation.checkpoints`                | list[object] | Ordered reward checkpoints (S-axis and V-axis)                                          |
| `process_evaluation.checkpoints[].id`           | string       | Descriptive checkpoint identifier                                                       |
| `process_evaluation.checkpoints[].axis`         | string       | `S` (strategy) or `V` (visual)                                                      |
| `process_evaluation.checkpoints[].tools`        | list[string] | Expected tool(s) for this step (`python_image_processing` or `google_search`, etc.) |
| `process_evaluation.checkpoints[].code_check`   | object       | AST verifier (tool + verifier function name)                                            |
| `process_evaluation.checkpoints[].visual_check` | object       | VLM judge check (question + expected answer)                                            |
| `process_evaluation.checkpoints[].search_check` | object       | Search quality verification (keywords + expected content)                               |
| `process_evaluation.checkpoints[].answer_check` | object       | Match verification (match_type + target)                                                |
| `process_evaluation.ordering_constraints`       | list[object] | Required checkpoint sequences with max_gap                                              |
| `process_evaluation.efficiency`                 | object       | reference_tool_calls, max_tool_calls, penalty_per_extra_call                            |

---

## Quick Start

```python
from datasets import load_dataset

# Load all environments
dataset = load_dataset("ethara/Janus", split="test")
print(f"{len(dataset)} environments across {len(set(dataset['meta']['domain']))} domains")

# Inspect an environment
env = dataset[0]
print(f"Task:        {env['task_id']}")
print(f"Level:       {env['meta']['level']}")
print(f"Domain:      {env['meta']['domain']}")
print(f"Checkpoints: {len(env['process_evaluation']['checkpoints'])}")
print(f"Question:    {env['input']['prompt'][:200]}...")

# Filter by domain
numismatics = dataset.filter(lambda x: x["meta"]["domain"] == "Numismatics")
print(f"{len(numismatics)} environments in Numismatics domain")
```

---

## Tool Interface

Janus environments expose a unified tool interface supporting both sandboxed code execution and structured function-calling APIs.

### Visual Tools (13 operations)

| Tool         | Description                                          |
| ------------ | ---------------------------------------------------- |
| crop         | Extract a region via bounding box with optional zoom |
| rotate       | Rotate image by angle (positive = counterclockwise)  |
| flip         | Mirror horizontally, vertically, or both             |
| resize       | Resize by target dimensions or scale factor          |
| enhance      | Adjust contrast/brightness for readability           |
| sharpen      | Apply sharpening filter                              |
| denoise      | Remove noise from image                              |
| autocontrast | Automatically maximize contrast range                |
| blur         | Apply Gaussian blur                                  |
| edge_detect  | Detect edges in image                                |
| threshold    | Binary threshold segmentation                        |
| equalize     | Histogram equalization for better contrast           |
| invert       | Invert image colors                                  |

### Retrieval Tools (4 operations)

| Tool               | Description                                  |
| ------------------ | -------------------------------------------- |
| google_search      | Text search returning top results            |
| google_lens_search | Reverse image search                         |
| fetch_webpage      | Webpage-to-text extraction                   |
| download_image     | Download external image into local workspace |

---

## RL Training Protocol

Janus supports training via the following reward structure:

| Step | Action                      | Reward Signal                                                                  |
| ---- | --------------------------- | ------------------------------------------------------------------------------ |
| 1    | Agent selects tool          | S-axis: correct tool selection (+1) or wrong tool (0)                          |
| 2    | Agent executes tool         | V-axis: artifact contains evidence (+1) or not (0)                             |
| 3    | Agent retrieves knowledge   | S-axis: retrieved content matches expected intermediate answer (+1) or not (0) |
| 4    | Agent produces final answer | Accuracy: correct (+1) or incorrect (0)                                        |
| End  | Episode complete            | Overthink penalty applied to total reward                                      |

### AST Verifiers

Process-level code checks use AST verification to confirm tool execution without relying on output inspection:

| Verifier               | Validates                                               |
| ---------------------- | ------------------------------------------------------- |
| `verify_crop_ast`    | Agent executed a crop operation with valid bounding box |
| `verify_rotate_ast`  | Agent executed a rotation with valid angle              |
| `verify_resize_ast`  | Agent executed a resize with valid dimensions           |
| `verify_enhance_ast` | Agent executed contrast/brightness enhancement          |
| `verify_flip_ast`    | Agent executed a flip operation                         |

---

## Links

| Resource                | Link                                                       |
| ----------------------- | ---------------------------------------------------------- |
| 📄 Reference Paper      | [arXiv:2604.03016](https://arxiv.org/abs/2604.03016)          |
| 💻 Hugging Face Dataset | [janus dataset](https://huggingface.co/datasets/ethara/Janus) |

---

## License

Janus is released under the **[Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (CC-BY-NC-ND 4.0)](https://creativecommons.org/licenses/by-nc-nd/4.0/)** license.

### Attribution Chain

| Layer                | Work                                                                    | License         | Required Attribution                                    |
| -------------------- | ----------------------------------------------------------------------- | --------------- | ------------------------------------------------------- |
| Upstream methodology | Agentic-MME (Wei et al., 2026)                                          | CC-BY 4.0       | Cite the paper in any publication; preserve this notice |
| Tool implementations | Visual and retrieval tool APIs                                          | MIT             | Each tool's attribution notice applies                  |
| Janus contribution   | RL environment curation, reward signal design, human-curated references | CC-BY-NC-ND 4.0 | Cite Janus and Agentic-MME                              |

### What You Can Do

- Use for research and education
- Share verbatim copies with proper attribution

### What You Must Do

- Give appropriate credit to both Janus and the upstream Agentic-MME paper
- Provide a link to the CC-BY-NC-ND 4.0 license
- Indicate if changes were made
- Preserve upstream licenses for any redistributed components

### What You Cannot Do

- Use for commercial purposes without separate permission
- Distribute modified or derivative versions of this dataset
- Sub-license the material

---

<div align="center">
  <sub>Janus. Process-level RL environments for multimodal agentic intelligence.</sub>
</div>
