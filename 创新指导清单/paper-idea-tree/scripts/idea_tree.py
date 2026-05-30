#!/usr/bin/env python3
"""Maintain a weighted paper idea tree."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_JSON = "idea_tree.json"
DEFAULT_MD = "创意树.md"
ASCII_MD_ALIAS = "idea_tree.md"
BINDINGS_JSON = Path(__file__).resolve().parents[1] / "bindings.json"


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def clamp_score(value: float | None) -> float:
    if value is None:
        return 0.0
    return max(0.0, min(5.0, float(value)))


def compute_weight(factors: dict[str, Any]) -> float:
    own_gain = clamp_score(factors.get("own_gain"))
    paper_gain = clamp_score(factors.get("paper_gain"))
    novelty = clamp_score(factors.get("novelty"))
    compatibility = clamp_score(factors.get("compatibility"))
    cost = clamp_score(factors.get("cost"))
    confidence = clamp_score(factors.get("confidence"))
    score = (
        0.30 * own_gain
        + 0.25 * paper_gain
        + 0.15 * novelty
        + 0.15 * compatibility
        + 0.15 * confidence
        - 0.15 * cost
    )
    return round(score, 3)


def load_bindings() -> dict[str, Any]:
    if not BINDINGS_JSON.exists():
        return {"projects": {}}
    with BINDINGS_JSON.open("r", encoding="utf-8") as f:
        data = json.load(f)
    data.setdefault("projects", {})
    return data


def resolve_tree_root(args: argparse.Namespace) -> Path:
    if getattr(args, "root", None):
        return Path(args.root).resolve()

    bindings = load_bindings()
    project_name = getattr(args, "project", None) or bindings.get("default_project")
    if project_name:
        project = bindings.get("projects", {}).get(project_name)
        if not project:
            known = ", ".join(sorted(bindings.get("projects", {}))) or "none"
            raise SystemExit(f"Unknown project binding: {project_name}. Known: {known}")
        tree_root = project.get("tree_root")
        if not tree_root:
            raise SystemExit(f"Project binding has no tree_root: {project_name}")
        return Path(tree_root).resolve()

    return Path(".").resolve()


def project_metadata(args: argparse.Namespace) -> dict[str, Any] | None:
    bindings = load_bindings()
    project_name = getattr(args, "project", None) or bindings.get("default_project")
    if not project_name:
        return None
    project = bindings.get("projects", {}).get(project_name)
    if not project:
        return None
    return {
        "name": project_name,
        "project_root": project.get("project_root", ""),
        "tree_root": project.get("tree_root", ""),
        "notes": project.get("notes", ""),
    }


def tree_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    root_path = resolve_tree_root(args)
    json_name = getattr(args, "json", DEFAULT_JSON)
    md_name = getattr(args, "md", DEFAULT_MD)
    return root_path / json_name, root_path / md_name


def empty_tree() -> dict[str, Any]:
    return {
        "version": 1,
        "updated_at": now(),
        "papers": [],
        "nodes": [],
    }


def load_tree(path: Path) -> dict[str, Any]:
    if not path.exists():
        return empty_tree()
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    data.setdefault("version", 1)
    data.setdefault("papers", [])
    data.setdefault("nodes", [])
    return data


def save_tree(path: Path, data: dict[str, Any]) -> None:
    data["updated_at"] = now()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def find_by_title(items: list[dict[str, Any]], title: str) -> dict[str, Any] | None:
    target = title.strip().lower()
    for item in items:
        if str(item.get("title", "")).strip().lower() == target:
            return item
    return None


def find_node(data: dict[str, Any], node_id: str) -> dict[str, Any] | None:
    for node in data["nodes"]:
        if node.get("id") == node_id:
            return node
    return None


def add_history(node: dict[str, Any], note: str | None) -> None:
    if note:
        node.setdefault("history", []).append({"at": now(), "note": note})


def add_paper(args: argparse.Namespace) -> None:
    json_path, md_path = tree_paths(args)
    data = load_tree(json_path)
    meta = project_metadata(args)
    if meta:
        data["project"] = meta
    paper = find_by_title(data["papers"], args.title)
    payload = {
        "title": args.title,
        "file": args.file or "",
        "venue": args.venue or "",
        "year": args.year or "",
        "summary": args.summary or "",
        "tasks": split_csv(args.tasks),
        "datasets": split_csv(args.datasets),
        "notes": args.notes or "",
    }
    if paper:
        paper.update({k: v for k, v in payload.items() if v not in ("", [], None)})
    else:
        data["papers"].append(payload)
    save_tree(json_path, data)
    render_file(data, md_path)
    print(f"Saved paper: {args.title}")


def add_node(args: argparse.Namespace) -> None:
    json_path, md_path = tree_paths(args)
    data = load_tree(json_path)
    meta = project_metadata(args)
    if meta:
        data["project"] = meta
    node = find_node(data, args.id)
    if node is None:
        node = {
            "id": args.id,
            "title": args.title,
            "category": args.category or "other",
            "summary": args.summary or "",
            "mechanism": args.mechanism or "",
            "why_good": args.why_good or "",
            "source_papers": [],
            "metrics": [],
            "tags": split_csv(args.tags),
            "connections": split_csv(args.connections),
            "status": args.status or "candidate",
            "weight_factors": {
                "paper_gain": 0,
                "own_gain": 0,
                "novelty": 0,
                "compatibility": 0,
                "cost": 0,
                "confidence": 0,
            },
            "weight": 0,
            "history": [],
        }
        data["nodes"].append(node)
    else:
        for key in ("title", "category", "summary", "mechanism", "why_good", "status"):
            value = getattr(args, key, None)
            if value:
                node[key] = value
        if args.tags:
            node["tags"] = sorted(set(node.get("tags", []) + split_csv(args.tags)))
        if args.connections:
            node["connections"] = sorted(set(node.get("connections", []) + split_csv(args.connections)))

    if args.paper:
        source = {"title": args.paper, "evidence": args.evidence or ""}
        if source not in node.setdefault("source_papers", []):
            node["source_papers"].append(source)
    add_history(node, args.note)
    node["weight"] = compute_weight(node.get("weight_factors", {}))
    save_tree(json_path, data)
    render_file(data, md_path)
    print(f"Saved node: {args.id}")


def update_node(args: argparse.Namespace) -> None:
    json_path, md_path = tree_paths(args)
    data = load_tree(json_path)
    meta = project_metadata(args)
    if meta:
        data["project"] = meta
    node = find_node(data, args.id)
    if node is None:
        raise SystemExit(f"Node not found: {args.id}")
    factors = node.setdefault("weight_factors", {})
    for key in ("paper_gain", "own_gain", "novelty", "compatibility", "cost", "confidence"):
        value = getattr(args, key)
        if value is not None:
            factors[key] = clamp_score(value)
    if args.status:
        node["status"] = args.status
    if args.metric:
        for metric in args.metric:
            node.setdefault("metrics", []).append({"source": "user-experiment", "note": metric})
    add_history(node, args.note)
    node["weight"] = compute_weight(factors)
    save_tree(json_path, data)
    render_file(data, md_path)
    print(f"Updated node: {args.id} weight={node['weight']}")


def render_markdown(data: dict[str, Any]) -> str:
    nodes = sorted(data.get("nodes", []), key=lambda n: (n.get("weight", 0), n.get("title", "")), reverse=True)
    lines = [
        "# 创意树",
        "",
        f"更新时间：{data.get('updated_at', '')}",
    ]
    project = data.get("project")
    if project:
        lines.extend(
            [
                "",
                f"绑定项目：{project.get('name', '')}",
                f"项目路径：`{project.get('project_root', '')}`",
            ]
        )
        if project.get("notes"):
            lines.append(f"说明：{project.get('notes')}")
    lines.extend(
        [
        "",
        "## 排名",
        "",
        "| 排名 | 权重 | 状态 | 节点 | 类别 | 核心理由 |",
        "|---:|---:|---|---|---|---|",
        ]
    )
    for idx, node in enumerate(nodes, 1):
        reason = str(node.get("why_good") or node.get("summary") or "").replace("|", "/")
        lines.append(
            f"| {idx} | {node.get('weight', 0)} | {node.get('status', '')} | "
            f"{node.get('title', node.get('id', ''))} | {node.get('category', '')} | {reason} |"
        )

    lines.extend(["", "## 节点详情", ""])
    for node in nodes:
        lines.extend(
            [
                f"### {node.get('title', node.get('id', ''))}",
                "",
                f"- ID: `{node.get('id', '')}`",
                f"- 权重: `{node.get('weight', 0)}`",
                f"- 状态: `{node.get('status', '')}`",
                f"- 类别: `{node.get('category', '')}`",
                f"- 摘要: {node.get('summary', '')}",
                f"- 机制: {node.get('mechanism', '')}",
                f"- 好在哪里: {node.get('why_good', '')}",
            ]
        )
        factors = node.get("weight_factors", {})
        if factors:
            factor_text = ", ".join(f"{k}={v}" for k, v in factors.items())
            lines.append(f"- 评分因子: {factor_text}")
        tags = node.get("tags", [])
        if tags:
            lines.append(f"- 标签: {', '.join(tags)}")
        sources = node.get("source_papers", [])
        if sources:
            lines.append("- 来源论文:")
            for source in sources:
                evidence = source.get("evidence", "")
                suffix = f"；证据: {evidence}" if evidence else ""
                lines.append(f"  - {source.get('title', '')}{suffix}")
        metrics = node.get("metrics", [])
        if metrics:
            lines.append("- 数据/实验:")
            for metric in metrics:
                if set(metric.keys()) <= {"source", "note"}:
                    lines.append(f"  - [{metric.get('source', '')}] {metric.get('note', '')}")
                else:
                    lines.append(
                        "  - "
                        f"{metric.get('dataset', '')} {metric.get('setting', '')} "
                        f"{metric.get('metric', '')}: {metric.get('value', '')} "
                        f"gain {metric.get('gain', '')} ({metric.get('source', '')}) "
                        f"{metric.get('note', '')}"
                    )
        history = node.get("history", [])
        if history:
            lines.append("- 更新记录:")
            for item in history[-5:]:
                lines.append(f"  - {item.get('at', '')}: {item.get('note', '')}")
        lines.append("")

    lines.extend(["## 论文索引", ""])
    for paper in data.get("papers", []):
        venue = " ".join(str(x) for x in [paper.get("venue", ""), paper.get("year", "")] if x)
        lines.append(f"- {paper.get('title', '')} ({venue}): {paper.get('summary', '')}")
    lines.append("")
    return "\n".join(lines)


def render_file(data: dict[str, Any], md_path: Path) -> None:
    md_path.parent.mkdir(parents=True, exist_ok=True)
    content = render_markdown(data)
    md_path.write_text(content, encoding="utf-8")
    if md_path.name == DEFAULT_MD:
        (md_path.parent / ASCII_MD_ALIAS).write_text(content, encoding="utf-8")


def init(args: argparse.Namespace) -> None:
    json_path, md_path = tree_paths(args)
    data = load_tree(json_path)
    meta = project_metadata(args)
    if meta:
        data["project"] = meta
    save_tree(json_path, data)
    render_file(data, md_path)
    print(f"Tree ready: {json_path}")
    print(f"Rendered: {md_path}")


def render(args: argparse.Namespace) -> None:
    json_path, md_path = tree_paths(args)
    data = load_tree(json_path)
    meta = project_metadata(args)
    if meta:
        data["project"] = meta
    for node in data.get("nodes", []):
        node["weight"] = compute_weight(node.get("weight_factors", {}))
    save_tree(json_path, data)
    render_file(data, md_path)
    print(f"Rendered: {md_path}")


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root", help="Workspace root. Overrides project binding when set.")
    parser.add_argument("--project", help="Project binding name, e.g. DVSR.")
    parser.add_argument("--json", default=DEFAULT_JSON, help="JSON filename.")
    parser.add_argument("--md", default=DEFAULT_MD, help="Markdown filename.")


def show_bindings(args: argparse.Namespace) -> None:
    print(json.dumps(load_bindings(), ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("init")
    add_common(p)
    p.set_defaults(func=init)

    p = sub.add_parser("render")
    add_common(p)
    p.set_defaults(func=render)

    p = sub.add_parser("bindings")
    p.set_defaults(func=show_bindings)

    p = sub.add_parser("add-paper")
    add_common(p)
    p.add_argument("--title", required=True)
    p.add_argument("--file")
    p.add_argument("--venue")
    p.add_argument("--year")
    p.add_argument("--summary")
    p.add_argument("--tasks")
    p.add_argument("--datasets")
    p.add_argument("--notes")
    p.set_defaults(func=add_paper)

    p = sub.add_parser("add-node")
    add_common(p)
    p.add_argument("--id", required=True)
    p.add_argument("--title", required=True)
    p.add_argument("--category")
    p.add_argument("--summary")
    p.add_argument("--mechanism")
    p.add_argument("--why-good", dest="why_good")
    p.add_argument("--paper")
    p.add_argument("--evidence")
    p.add_argument("--tags")
    p.add_argument("--connections")
    p.add_argument("--status")
    p.add_argument("--note")
    p.set_defaults(func=add_node)

    p = sub.add_parser("update-node")
    add_common(p)
    p.add_argument("--id", required=True)
    p.add_argument("--paper-gain", type=float)
    p.add_argument("--own-gain", type=float)
    p.add_argument("--novelty", type=float)
    p.add_argument("--compatibility", type=float)
    p.add_argument("--cost", type=float)
    p.add_argument("--confidence", type=float)
    p.add_argument("--status")
    p.add_argument("--metric", action="append")
    p.add_argument("--note")
    p.set_defaults(func=update_node)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
