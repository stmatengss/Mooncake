#!/usr/bin/env python3

import json
import os
import subprocess
import sys
from pathlib import Path


ENGLISH_ACTION_KEYWORDS = {
    "implement",
    "add",
    "create",
    "build",
    "develop",
    "introduce",
    "support",
    "enable",
}

ENGLISH_TARGET_KEYWORDS = {
    "feature",
    "api",
    "capability",
    "workflow",
    "integration",
}

CHINESE_ACTION_KEYWORDS = {
    "新增",
    "添加",
    "实现",
    "开发",
    "引入",
    "支持",
    "新功能",
    "需求实现",
}

CHINESE_TARGET_KEYWORDS = {
    "功能",
    "接口",
    "能力",
    "流程",
    "工作流",
    "支持",
    "集成",
    "API",
    "api",
}


def load_input() -> dict:
    try:
        return json.load(sys.stdin)
    except json.JSONDecodeError:
        return {}


def looks_like_new_feature(prompt: str) -> bool:
    normalized = " ".join(prompt.split())
    lowered = normalized.lower()

    if "new feature" in lowered or "feature request" in lowered:
        return True

    english_action = any(keyword in lowered for keyword in ENGLISH_ACTION_KEYWORDS)
    english_target = any(keyword in lowered for keyword in ENGLISH_TARGET_KEYWORDS)
    chinese_action = any(keyword in normalized for keyword in CHINESE_ACTION_KEYWORDS)
    chinese_target = any(keyword in normalized for keyword in CHINESE_TARGET_KEYWORDS)
    return (english_action and english_target) or (chinese_action and chinese_target)


def git_output(repo_root: str, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", repo_root, *args],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def is_linked_worktree(repo_root: str) -> bool:
    git_entry = Path(repo_root) / ".git"
    return git_entry.is_file()


def current_branch(repo_root: str) -> str:
    return git_output(repo_root, "branch", "--show-current")


def default_branch(repo_root: str) -> str:
    symbolic = git_output(repo_root, "symbolic-ref", "refs/remotes/origin/HEAD")
    if symbolic.startswith("refs/remotes/origin/"):
        return symbolic.removeprefix("refs/remotes/origin/")
    return "main"


def fail(message: str) -> None:
    print(message, file=sys.stderr)
    sys.exit(2)


def main() -> int:
    payload = load_input()
    prompt = payload.get("prompt", "")
    if not prompt or not looks_like_new_feature(prompt):
        return 0

    repo_root = git_output(os.getcwd(), "rev-parse", "--show-toplevel")
    if not repo_root:
        return 0

    branch = current_branch(repo_root) or "(detached)"
    base_branch = default_branch(repo_root)

    if not is_linked_worktree(repo_root):
        fail(
            "检测到这是新功能实现请求。请先用 git worktree 在独立分支中开始工作，再重新提交请求。\n"
            f"建议命令：git worktree add -b <branch-name> ../.worktrees/Mooncake/<branch-name> {base_branch}\n"
            f"当前分支：{branch}"
        )

    if branch in {"main", "master", base_branch}:
        fail(
            "检测到这是新功能实现请求，但当前仍在共享基础分支上。请先创建新的 worktree 分支，再重新提交请求。\n"
            f"建议命令：git worktree add -b <branch-name> ../.worktrees/Mooncake/<branch-name> {base_branch}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())