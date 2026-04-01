"""RLM TUI — interactive sessions, replay, and inspection views."""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.theme import Theme
from textual.widgets import (
    Collapsible,
    Footer,
    Header,
    Input,
    Label,
    RichLog,
    Static,
    Tree,
)
from textual.widgets._tree import TreeNode


# ──────────────────────────────────────────────
# Themes (from verifiers TUI)
# ──────────────────────────────────────────────

BLACK_WARM = Theme(
    name="black-warm",
    primary="#d4a373",
    secondary="#808080",
    accent="#c9ada7",
    warning="#ffa500",
    error="#ff6b6b",
    success="#98c379",
    background="#141414",
    surface="#141414",
    panel="#141414",
    foreground="#ffffff",
    dark=True,
)

WHITE_WARM = Theme(
    name="white-warm",
    primary="#8b6f47",
    secondary="#606060",
    accent="#a08b87",
    warning="#ff8c00",
    error="#dc143c",
    success="#6b8e23",
    background="#f5f5f5",
    surface="#f5f5f5",
    panel="#f5f5f5",
    foreground="#1a1a1a",
    dark=False,
)


# ──────────────────────────────────────────────
# SessionData — reads session directories
# ──────────────────────────────────────────────

@dataclass
class SessionData:
    """Lightweight reader/tailer for a session directory."""

    path: Path
    meta: dict = field(default_factory=dict)
    entries: list[dict] = field(default_factory=list)
    children: dict[str, "SessionData"] = field(default_factory=dict)
    _offset: int = 0

    def load_all(self) -> None:
        """Read entire session (for replay / inspection)."""
        meta_path = self.path / "meta.json"
        if meta_path.exists():
            self.meta = json.loads(meta_path.read_text())

        msg_path = self.path / "messages.jsonl"
        if msg_path.exists():
            with open(msg_path) as f:
                self.entries = [json.loads(line) for line in f if line.strip()]
            self._offset = msg_path.stat().st_size

    def poll(self) -> list[dict]:
        """Read new lines since last poll; scan for new sub-* dirs."""
        new_entries: list[dict] = []

        # Re-read meta
        meta_path = self.path / "meta.json"
        if meta_path.exists():
            try:
                self.meta = json.loads(meta_path.read_text())
            except (json.JSONDecodeError, OSError):
                pass

        # Tail messages.jsonl
        msg_path = self.path / "messages.jsonl"
        if msg_path.exists():
            try:
                with open(msg_path) as f:
                    f.seek(self._offset)
                    raw = f.read()
                    self._offset = f.tell()
                for line in raw.splitlines():
                    if line.strip():
                        entry = json.loads(line)
                        self.entries.append(entry)
                        new_entries.append(entry)
            except (json.JSONDecodeError, OSError):
                pass

        # Discover new child dirs
        for child_dir in sorted(self.path.iterdir()):
            if child_dir.is_dir() and child_dir.name.startswith("sub-"):
                if child_dir.name not in self.children:
                    child = SessionData(path=child_dir)
                    self.children[child_dir.name] = child

        # Poll children
        for child in self.children.values():
            child.poll()

        return new_entries

    def load_tree(self) -> None:
        """Recursively load self + all children."""
        self.load_all()
        for child_dir in sorted(self.path.iterdir()):
            if child_dir.is_dir() and child_dir.name.startswith("sub-"):
                child = SessionData(path=child_dir)
                child.load_tree()
                self.children[child_dir.name] = child

    def all_entries_sorted(self) -> list[tuple[str, dict]]:
        """Flatten all entries across tree, sorted by timestamp.
        Returns (session_name, entry) tuples."""
        result: list[tuple[str, dict]] = []
        name = self.meta.get("session_id", self.path.name)
        for entry in self.entries:
            result.append((name, entry))
        for child_name, child in self.children.items():
            for item in child.all_entries_sorted():
                result.append(item)
        result.sort(key=lambda x: x[1].get("timestamp", 0))
        return result


def _extract_tool_call(tc: dict) -> tuple[str, str]:
    """Extract (name, args_raw) from a tool call dict.

    Handles both the engine log format {"name": ..., "args": ...}
    and the OpenAI wire format {"function": {"name": ..., "arguments": ...}}.
    """
    if "function" in tc:
        fn = tc["function"]
        return fn.get("name", "?"), fn.get("arguments", "")
    return tc.get("name", "?"), str(tc.get("args", ""))


# ──────────────────────────────────────────────
# SessionPane — RichLog for one session (split-pane view)
# ──────────────────────────────────────────────

class SessionPane(RichLog):
    """Renders one session's message stream."""

    def __init__(self, session_data: SessionData, **kwargs):
        super().__init__(highlight=True, markup=False, wrap=True, **kwargs)
        self.session_data = session_data
        self._entry_count = 0

    def on_mount(self) -> None:
        meta = self.session_data.meta
        sid = meta.get("session_id", self.session_data.path.name)[:12]
        model = meta.get("model", "?")
        status = meta.get("status", "?")
        header = Text()
        header.append(f" {sid} ", style="bold reverse")
        header.append(f" {model} ", style="dim")
        header.append(f" {status} ", style="bold")
        self.write(header)
        self.write(Text("─" * 40, style="dim"))

    def append_entry(self, entry: dict) -> None:
        self._entry_count += 1
        etype = entry.get("type", "")

        if etype == "assistant":
            self._render_assistant(entry)
        elif etype == "tool_result":
            self._render_tool_result(entry)
        elif etype == "sub_spawn":
            self._render_sub_spawn(entry)
        elif etype == "done":
            self._render_done(entry)

    def _render_assistant(self, entry: dict) -> None:
        turn = entry.get("turn", "?")
        tool_calls = entry.get("tool_calls")
        content = entry.get("content")

        if tool_calls:
            for tc in tool_calls:
                name, args_raw = _extract_tool_call(tc)
                if isinstance(args_raw, str) and len(args_raw) > 120:
                    args_raw = args_raw[:120] + "..."
                line = Text()
                line.append(f"[{turn}] ", style="dim")
                line.append(f"{name}", style="bold #ffa500")
                line.append(f"({args_raw})", style="dim")
                self.write(line)
        if content:
            line = Text()
            line.append(f"[{turn}] ", style="dim")
            line.append(content[:500], style="#98c379")
            self.write(line)

    def _render_tool_result(self, entry: dict) -> None:
        tool = entry.get("tool", "?")
        content = entry.get("content", "")
        duration = entry.get("duration", 0)

        line = Text()
        line.append(f"  ↳ {tool}", style="bold dim")
        line.append(f" ({duration:.1f}s)", style="dim")
        self.write(line)

        if content:
            for output_line in content.splitlines()[:15]:
                out = Text()
                out.append(f"    {output_line}", style="dim")
                self.write(out)
            if content.count("\n") > 15:
                self.write(Text("    ...", style="dim"))

    def _render_sub_spawn(self, entry: dict) -> None:
        child = entry.get("child_dir", "?")
        cmd = entry.get("command", "")
        line = Text()
        line.append("  ⤷ ", style="bold #c9ada7")
        line.append(f"spawn {child}", style="bold #c9ada7")
        if cmd:
            line.append(f"  {cmd[:80]}", style="dim")
        self.write(line)

    def _render_done(self, entry: dict) -> None:
        answer = entry.get("answer", "")
        line = Text()
        line.append(" ✓ ", style="bold #98c379")
        line.append(answer[:300], style="#98c379")
        self.write(line)
        self.write(Text("─" * 40, style="dim"))


# ──────────────────────────────────────────────
# PaneLayout — recursive split-pane columns
# ──────────────────────────────────────────────

class PaneLayout(Container):
    """Manages nested split-pane layout by recursive depth."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._panes: dict[str, SessionPane] = {}

    def rebuild(self, root: SessionData) -> None:
        """Clear and remount panes to reflect the session tree."""
        self.remove_children()
        self._panes.clear()
        columns = self._collect_by_depth(root, depth=0)
        self._mount_columns(columns)

    def _collect_by_depth(
        self, sd: SessionData, depth: int
    ) -> dict[int, list[SessionData]]:
        result: dict[int, list[SessionData]] = {}
        result.setdefault(depth, []).append(sd)
        for child in sd.children.values():
            child_cols = self._collect_by_depth(child, depth + 1)
            for d, sessions in child_cols.items():
                result.setdefault(d, []).extend(sessions)
        return result

    def _mount_columns(self, columns: dict[int, list[SessionData]]) -> None:
        max_depth = min(max(columns.keys(), default=0), 2)  # max 3 columns (0,1,2)

        if max_depth == 0:
            # Single pane
            sd = columns[0][0]
            pane = SessionPane(sd, id=f"pane-{sd.path.name}")
            self._panes[sd.path.name] = pane
            self.mount(pane)
            return

        # Multiple columns
        for depth in range(max_depth + 1):
            sessions = columns.get(depth, [])
            if len(sessions) == 1:
                sd = sessions[0]
                pane = SessionPane(sd, id=f"pane-{sd.path.name}")
                self._panes[sd.path.name] = pane
                self.mount(pane)
            else:
                panes = []
                for sd in sessions:
                    pane = SessionPane(sd, id=f"pane-{sd.path.name}")
                    self._panes[sd.path.name] = pane
                    panes.append(pane)
                self.mount(Vertical(*panes))

    def get_pane(self, session_name: str) -> SessionPane | None:
        return self._panes.get(session_name)


# ──────────────────────────────────────────────
# SectionData + InspectionView — collapsible overview
# ──────────────────────────────────────────────

@dataclass(frozen=True)
class SectionData:
    """One collapsible section in the inspection view."""

    title: str
    body: str
    collapsed: bool
    classes: str
    nested_sections: tuple[SectionData, ...] = ()


def _build_sections(sd: SessionData) -> list[SectionData]:
    """Convert a session tree into collapsible SectionData list."""
    sections: list[SectionData] = []
    i = 0
    entries = sd.entries

    while i < len(entries):
        entry = entries[i]
        etype = entry.get("type", "")

        if etype == "assistant":
            tool_calls = entry.get("tool_calls")
            content = entry.get("content")
            turn = entry.get("turn", "?")

            if tool_calls:
                # Gather following tool_result entries
                nested: list[SectionData] = []
                for tc in tool_calls:
                    name, args_raw = _extract_tool_call(tc)
                    if isinstance(args_raw, str) and len(args_raw) > 200:
                        args_raw = args_raw[:200] + "..."

                    # Find matching tool_result
                    result_body = ""
                    if i + 1 < len(entries) and entries[i + 1].get("type") == "tool_result":
                        i += 1
                        result_entry = entries[i]
                        dur = result_entry.get("duration", 0)
                        result_body = result_entry.get("content", "")
                        nested.append(SectionData(
                            title=f"↳ {name} ({dur:.1f}s)",
                            body=result_body,
                            collapsed=True,
                            classes="history-section tool-section nested-section",
                        ))
                    else:
                        nested.append(SectionData(
                            title=f"↳ {name}",
                            body=args_raw,
                            collapsed=True,
                            classes="history-section tool-section nested-section",
                        ))

                if content:
                    body = content[:500]
                else:
                    body = ""

                sections.append(SectionData(
                    title=f"Turn {turn} — assistant (tools)",
                    body=body,
                    collapsed=False,
                    classes="history-section assistant-section",
                    nested_sections=tuple(nested),
                ))

            elif content:
                sections.append(SectionData(
                    title=f"Turn {turn} — assistant",
                    body=content[:500],
                    collapsed=False,
                    classes="history-section assistant-section",
                ))

        elif etype == "tool_result":
            # Orphaned tool result (not paired with assistant)
            tool = entry.get("tool", "?")
            dur = entry.get("duration", 0)
            sections.append(SectionData(
                title=f"↳ {tool} ({dur:.1f}s)",
                body=entry.get("content", ""),
                collapsed=True,
                classes="history-section tool-section",
            ))

        elif etype == "sub_spawn":
            child_dir_name = entry.get("child_dir", "")
            cmd = entry.get("command", "")
            # Find child session data and recurse
            child_sd = sd.children.get(child_dir_name)
            if child_sd:
                child_sections = _build_sections(child_sd)
                nested_tup = tuple(
                    SectionData(
                        title=s.title,
                        body=s.body,
                        collapsed=s.collapsed,
                        classes=s.classes + " nested-section",
                        nested_sections=s.nested_sections,
                    )
                    for s in child_sections
                )
            else:
                nested_tup = ()
            sections.append(SectionData(
                title=f"⤷ sub-agent: {cmd[:80]}" if cmd else f"⤷ sub-agent {child_dir_name}",
                body="",
                collapsed=False,
                classes="history-section sub-spawn-section",
                nested_sections=nested_tup,
            ))

        elif etype == "done":
            answer = entry.get("answer", "")
            turns = entry.get("turns", "?")
            sections.append(SectionData(
                title=f"✓ Done ({turns} turns)",
                body=answer,
                collapsed=False,
                classes="history-section done-section",
            ))

        i += 1

    return sections


def _make_collapsible(section: SectionData) -> Collapsible:
    """Build a Collapsible widget from SectionData (recursive)."""
    children: list[Any] = []
    if section.body:
        children.append(Static(section.body, classes="section-body", markup=False))
    for nested in section.nested_sections:
        children.append(_make_collapsible(nested))
    return Collapsible(
        *children,
        title=section.title,
        collapsed=section.collapsed,
        classes=section.classes,
    )


class InspectionView(Screen):
    """Verifiers-style expandable collapsible overview of a completed session."""

    BINDINGS = [
        Binding("q", "dismiss", "Back", show=True),
        Binding("escape", "dismiss", "Back"),
        Binding("e", "expand_all", "Expand all", show=True),
        Binding("x", "collapse_all", "Collapse all", show=True),
    ]

    def __init__(self, session_data: SessionData, **kwargs):
        super().__init__(**kwargs)
        self.session_data = session_data

    def compose(self) -> ComposeResult:
        meta = self.session_data.meta
        sid = meta.get("session_id", self.session_data.path.name)
        model = meta.get("model", "?")
        status = meta.get("status", "?")
        turns = meta.get("turns", "?")
        usage = meta.get("usage", {})

        summary = Text()
        summary.append(f" {sid} ", style="bold reverse")
        summary.append(f"  model={model}  turns={turns}  status={status}", style="dim")
        if usage:
            pt = usage.get("prompt_tokens", 0)
            ct = usage.get("completion_tokens", 0)
            summary.append(f"  tokens={pt}+{ct}", style="dim")

        yield Header()
        with Container(classes="metadata-panel"):
            with Horizontal(classes="metadata-layout"):
                yield Static(summary, id="metadata-summary")
        with VerticalScroll(id="inspection-scroll"):
            sections = _build_sections(self.session_data)
            for section in sections:
                yield _make_collapsible(section)
        yield Footer()

    @on(Collapsible.Expanded)
    def on_collapsible_expanded(self, event: Collapsible.Expanded) -> None:
        c = event.collapsible
        if not c.has_class("history-section"):
            return
        c.remove_class("expand-settle")
        c.add_class("just-expanded")
        self.set_timer(0.04, lambda: self._shift_expand_pulse(c))
        self.set_timer(0.10, lambda: self._clear_expand_pulse(c))
        c.call_after_refresh(
            lambda: c.scroll_visible(duration=0.06, easing="out_cubic")
        )

    def _shift_expand_pulse(self, c: Collapsible) -> None:
        if not c.is_mounted:
            return
        c.remove_class("just-expanded")
        c.add_class("expand-settle")

    def _clear_expand_pulse(self, c: Collapsible) -> None:
        if not c.is_mounted:
            return
        c.remove_class("just-expanded")
        c.remove_class("expand-settle")

    def action_expand_all(self) -> None:
        for section in self.query(Collapsible):
            section.collapsed = False

    def action_collapse_all(self) -> None:
        for section in self.query(Collapsible):
            section.collapsed = True


# ──────────────────────────────────────────────
# SessionBrowser — tree-based session picker
# ──────────────────────────────────────────────

@dataclass(frozen=True)
class BrowserNodeData:
    kind: str  # "root", "session", "child"
    path: Path | None = None


class SessionBrowser(Screen):
    """Browse and select sessions from ~/.rlm/sessions/."""

    BINDINGS = [
        Binding("q", "quit_app", "Quit", show=True),
        Binding("escape", "quit_app", "Quit"),
        Binding("enter", "select_session", "Open", show=True),
        Binding("i", "inspect_session", "Inspect", show=True),
    ]

    def __init__(self, sessions_dir: Path, **kwargs):
        super().__init__(**kwargs)
        self.sessions_dir = sessions_dir

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(classes="browser-panel"):
            with Container(classes="browser-tree-panel"):
                yield Label("Sessions", classes="title")
                tree: Tree[BrowserNodeData] = Tree("~/.rlm/sessions", id="session-tree")
                tree.root.data = BrowserNodeData(kind="root")
                self._populate_tree(tree)
                tree.root.expand()
                yield tree
            with Container(classes="browser-details-panel"):
                with VerticalScroll(id="browser-details-scroll"):
                    yield Static("Select a session to see details.", id="browser-details")
        yield Footer()

    def _populate_tree(self, tree: Tree[BrowserNodeData]) -> None:
        if not self.sessions_dir.exists():
            return
        for session_dir in sorted(self.sessions_dir.iterdir(), reverse=True):
            if not session_dir.is_dir():
                continue
            meta_path = session_dir / "meta.json"
            meta: dict = {}
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text())
                except (json.JSONDecodeError, OSError):
                    pass

            sid = meta.get("session_id", session_dir.name)[:12]
            status = meta.get("status", "?")
            prompt = meta.get("prompt_preview", "")[:40]
            turns = meta.get("turns", "?")

            icon = "✓" if status == "done" else "●" if status == "running" else "?"
            label = f"{icon} {sid}  {prompt}  ({turns}t)"

            node = tree.root.add(
                label,
                data=BrowserNodeData(kind="session", path=session_dir),
            )

            # Add sub-sessions as children
            for child_dir in sorted(session_dir.iterdir()):
                if child_dir.is_dir() and child_dir.name.startswith("sub-"):
                    child_meta: dict = {}
                    child_meta_path = child_dir / "meta.json"
                    if child_meta_path.exists():
                        try:
                            child_meta = json.loads(child_meta_path.read_text())
                        except (json.JSONDecodeError, OSError):
                            pass
                    child_status = child_meta.get("status", "?")
                    child_prompt = child_meta.get("prompt_preview", "")[:30]
                    child_icon = "✓" if child_status == "done" else "●"
                    node.add_leaf(
                        f"  {child_icon} {child_dir.name}  {child_prompt}",
                        data=BrowserNodeData(kind="child", path=child_dir),
                    )

    @on(Tree.NodeHighlighted)
    def on_tree_node_highlighted(self, event: Tree.NodeHighlighted) -> None:
        node_data = event.node.data
        if node_data is None or node_data.path is None:
            return
        meta_path = node_data.path / "meta.json"
        details_widget = self.query_one("#browser-details", Static)
        if not meta_path.exists():
            details_widget.update("No metadata found.")
            return
        try:
            meta = json.loads(meta_path.read_text())
        except (json.JSONDecodeError, OSError):
            details_widget.update("Error reading metadata.")
            return

        info = Text()
        info.append("Session Details\n", style="bold")
        info.append("─" * 30 + "\n", style="dim")
        for key in ["session_id", "model", "status", "depth", "turns", "cwd", "prompt_preview", "answer_preview"]:
            val = meta.get(key)
            if val is not None:
                info.append(f"{key}: ", style="bold")
                info.append(f"{val}\n")
        usage = meta.get("usage", {})
        if usage:
            info.append("\nUsage:\n", style="bold")
            for k, v in usage.items():
                info.append(f"  {k}: {v}\n", style="dim")
        details_widget.update(info)

    def _get_selected_path(self) -> Path | None:
        tree = self.query_one("#session-tree", Tree)
        node = tree.cursor_node
        if node is None or node.data is None:
            return None
        return node.data.path

    def action_select_session(self) -> None:
        path = self._get_selected_path()
        if path is not None:
            sd = SessionData(path=path)
            sd.load_tree()
            self.app.push_screen(ReplayView(sd))

    def action_inspect_session(self) -> None:
        path = self._get_selected_path()
        if path is not None:
            sd = SessionData(path=path)
            sd.load_tree()
            self.app.push_screen(InspectionView(sd))

    def action_quit_app(self) -> None:
        self.app.exit()


# ──────────────────────────────────────────────
# ReplayView — step-through replay with split panes
# ──────────────────────────────────────────────

class ReplayView(Screen):
    """Step-through replay of a completed session with nested panes."""

    BINDINGS = [
        Binding("q", "dismiss", "Back", show=True),
        Binding("escape", "dismiss", "Back"),
        Binding("j", "step_forward", "Next", show=True),
        Binding("space", "step_forward", "Next"),
        Binding("k", "step_back", "Prev", show=True),
        Binding("i", "inspect", "Inspect", show=True),
        Binding("z", "zoom", "Zoom", show=True),
        Binding("tab", "focus_next", "Next pane"),
    ]

    def __init__(self, session_data: SessionData, **kwargs):
        super().__init__(**kwargs)
        self.session_data = session_data
        self._flat_entries: list[tuple[str, dict]] = []
        self._cursor: int = 0

    def compose(self) -> ComposeResult:
        yield Header()
        yield PaneLayout(id="panes")

        meta = self.session_data.meta
        turns = meta.get("turns", "?")
        total = len(self.session_data.all_entries_sorted())
        yield Label(
            f" Replay: {self.session_data.path.name}  |  {total} entries  |  j/k step  |  i inspect ",
            classes="title",
        )
        yield Footer()

    def on_mount(self) -> None:
        self._flat_entries = self.session_data.all_entries_sorted()
        pane_layout = self.query_one("#panes", PaneLayout)
        pane_layout.rebuild(self.session_data)

    def action_step_forward(self) -> None:
        if self._cursor >= len(self._flat_entries):
            return
        session_name, entry = self._flat_entries[self._cursor]
        self._cursor += 1

        pane_layout = self.query_one("#panes", PaneLayout)
        pane = pane_layout.get_pane(session_name)
        if pane is None:
            # Try matching by path name
            for name, p in pane_layout._panes.items():
                if session_name in name or name in session_name:
                    pane = p
                    break
        if pane is not None:
            pane.append_entry(entry)

    def action_step_back(self) -> None:
        if self._cursor <= 0:
            return
        self._cursor -= 1

        # Rebuild all panes up to cursor
        pane_layout = self.query_one("#panes", PaneLayout)
        pane_layout.rebuild(self.session_data)

        # Replay entries up to cursor
        for idx in range(self._cursor):
            session_name, entry = self._flat_entries[idx]
            pane = pane_layout.get_pane(session_name)
            if pane is None:
                for name, p in pane_layout._panes.items():
                    if session_name in name or name in session_name:
                        pane = p
                        break
            if pane is not None:
                pane.append_entry(entry)

    def action_inspect(self) -> None:
        self.app.push_screen(InspectionView(self.session_data))

    def action_zoom(self) -> None:
        focused = self.focused
        if isinstance(focused, SessionPane):
            if focused.has_class("-zoomed"):
                # Unzoom: show everything
                focused.remove_class("-zoomed")
                for pane in self.query(SessionPane):
                    pane.remove_class("-hidden")
                for col in self.query("PaneLayout > Vertical"):
                    col.remove_class("-hidden")
            else:
                # Zoom: hide all other panes/columns
                for pane in self.query(SessionPane):
                    pane.remove_class("-zoomed")
                    if pane is not focused:
                        pane.add_class("-hidden")
                    else:
                        pane.remove_class("-hidden")
                for col in self.query("PaneLayout > Vertical"):
                    if focused not in col.query(SessionPane):
                        col.add_class("-hidden")
                    else:
                        col.remove_class("-hidden")
                focused.add_class("-zoomed")


# ──────────────────────────────────────────────
# RLMApp — main application
# ──────────────────────────────────────────────

class RLMApp(App):
    """RLM TUI application."""

    CSS_PATH = "tui.tcss"
    ENABLE_COMMAND_PALETTE = False

    BINDINGS = [
        Binding("q", "quit", "Quit", show=True),
        Binding("d", "toggle_dark", "Theme", show=True),
    ]

    def __init__(self, mode: str = "interactive", session_data: SessionData | None = None, **kwargs):
        super().__init__(**kwargs)
        self.mode = mode
        self.initial_session_data = session_data

    def on_mount(self) -> None:
        self.register_theme(BLACK_WARM)
        self.register_theme(WHITE_WARM)
        self.theme = "black-warm"

        if self.mode == "browse":
            sessions_dir = Path.home() / ".rlm" / "sessions"
            self.push_screen(SessionBrowser(sessions_dir))
        elif self.mode == "replay" and self.initial_session_data:
            self.push_screen(ReplayView(self.initial_session_data))
        elif self.mode == "inspect" and self.initial_session_data:
            self.push_screen(InspectionView(self.initial_session_data))
        else:
            self.push_screen(InteractiveView())

    def action_toggle_dark(self) -> None:
        if self.theme == "black-warm":
            self.theme = "white-warm"
        else:
            self.theme = "black-warm"


# ──────────────────────────────────────────────
# InteractiveView — live session with engine
# ──────────────────────────────────────────────

class InteractiveView(Screen):
    """Live interactive session: type prompts, watch the agent work."""

    BINDINGS = [
        Binding("escape", "abort", "Abort"),
        Binding("z", "zoom", "Zoom", show=True),
        Binding("tab", "focus_next", "Next pane"),
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._root_session: SessionData | None = None
        self._engine: Any = None
        self._poll_timer: Any = None
        self._queued_prompt: str | None = None
        self._engine_running = False

    def compose(self) -> ComposeResult:
        yield Header()
        yield PaneLayout(id="panes")
        yield Input(placeholder="Type your prompt...", id="prompt-input")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#prompt-input", Input).focus()

    @on(Input.Submitted, "#prompt-input")
    def on_prompt_submitted(self, event: Input.Submitted) -> None:
        prompt = event.value.strip()
        if not prompt:
            return
        event.input.clear()

        if self._engine_running:
            self._queued_prompt = prompt
            return

        self._start_run(prompt)

    def _start_run(self, prompt: str) -> None:
        from rlm.session import Session
        from rlm.engine import RLMEngine

        session = Session()
        self._root_session = SessionData(path=session.dir)

        pane_layout = self.query_one("#panes", PaneLayout)
        pane_layout.rebuild(self._root_session)

        try:
            self._engine = RLMEngine(session=session)
        except Exception as e:
            self._show_error(str(e))
            return

        self._engine_running = True
        self._run_engine_thread(prompt)
        self._poll_timer = self.set_interval(0.3, self._poll_sessions)

    @work(thread=True)
    def _run_engine_thread(self, prompt: str) -> None:
        asyncio.run(self._engine.run(prompt))

    def _poll_sessions(self) -> None:
        if self._root_session is None:
            return

        new_entries = self._root_session.poll()
        pane_layout = self.query_one("#panes", PaneLayout)

        # Check if tree structure changed (new children)
        needs_rebuild = self._check_new_children(pane_layout)
        if needs_rebuild:
            # Rebuild layout then replay all entries
            pane_layout.rebuild(self._root_session)
            self._replay_all(pane_layout)
        else:
            # Feed new entries to the root pane
            root_name = self._root_session.path.name
            pane = pane_layout.get_pane(root_name)
            if pane is not None:
                for entry in new_entries:
                    pane.append_entry(entry)

            # Also poll children and feed their entries
            self._poll_children(pane_layout, self._root_session)

    def _check_new_children(self, pane_layout: PaneLayout) -> bool:
        """Check if any new children appeared that aren't yet in the layout."""
        def _check(sd: SessionData) -> bool:
            for child_name in sd.children:
                if child_name not in pane_layout._panes and sd.children[child_name].path.name not in pane_layout._panes:
                    return True
                if _check(sd.children[child_name]):
                    return True
            return False
        return _check(self._root_session) if self._root_session else False

    def _poll_children(self, pane_layout: PaneLayout, sd: SessionData) -> None:
        for child_name, child_sd in sd.children.items():
            pane = pane_layout.get_pane(child_sd.path.name)
            if pane is not None:
                # Feed any un-rendered entries
                rendered = pane._entry_count
                for entry in child_sd.entries[rendered:]:
                    pane.append_entry(entry)
            self._poll_children(pane_layout, child_sd)

    def _replay_all(self, pane_layout: PaneLayout) -> None:
        """Replay all entries from the session tree into their panes."""
        def _replay(sd: SessionData) -> None:
            pane = pane_layout.get_pane(sd.path.name)
            if pane is not None:
                for entry in sd.entries:
                    pane.append_entry(entry)
            for child in sd.children.values():
                _replay(child)
        if self._root_session:
            _replay(self._root_session)

    def on_worker_state_changed(self, event) -> None:
        """Called when the engine worker finishes."""
        if not hasattr(event, "worker") or not event.worker.is_finished:
            return

        self._engine_running = False
        if self._poll_timer is not None:
            self._poll_timer.stop()
            self._poll_timer = None

        # Do one final poll to pick up any remaining entries
        self._poll_sessions()

        # Show error if the worker failed
        if event.worker.error is not None:
            self._show_error(str(event.worker.error))
            return

        if self._queued_prompt:
            prompt = self._queued_prompt
            self._queued_prompt = None
            self._start_run(prompt)

    def _show_error(self, error_msg: str) -> None:
        """Display an error message in the current pane or as a notification."""
        pane_layout = self.query_one("#panes", PaneLayout)
        root_name = self._root_session.path.name if self._root_session else None
        pane = pane_layout.get_pane(root_name) if root_name else None
        if pane is not None:
            line = Text()
            line.append(" ERROR ", style="bold reverse red")
            line.append(f" {error_msg}", style="red")
            pane.write(line)
        else:
            self.notify(f"Error: {error_msg}", severity="error")

    def action_abort(self) -> None:
        if self._engine_running:
            for worker in self.workers:
                worker.cancel()
            self._engine_running = False
            if self._poll_timer is not None:
                self._poll_timer.stop()
                self._poll_timer = None

    def action_zoom(self) -> None:
        focused = self.focused
        if isinstance(focused, SessionPane):
            if focused.has_class("-zoomed"):
                focused.remove_class("-zoomed")
            else:
                for pane in self.query(SessionPane):
                    pane.remove_class("-zoomed")
                focused.add_class("-zoomed")


# ──────────────────────────────────────────────
# Entry point for textual run --dev
# ──────────────────────────────────────────────

if __name__ == "__main__":
    app = RLMApp(mode="browse")
    app.run()
