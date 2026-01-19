#!/usr/bin/env python3
"""
MarkTerm - Beautiful terminal markdown reader with interactive clickable links
Clean rendering with no visible markdown syntax.

Installation:
    pip install textual markdown-it-py pygments rich Pillow httpx watchdog

Usage:
    python markterm.py document.md
"""

from __future__ import annotations
from utility import *
import os
import sys
import re
import subprocess
import time
from pathlib import Path
from typing import Optional
from io import BytesIO

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll, Horizontal, Vertical
from textual.widgets import Header, Footer, Static, Tree
from textual.reactive import reactive
from textual import work, events
from textual.message import Message

from markdown_it import MarkdownIt
from markdown_it.token import Token
from rich.text import Text
from rich.style import Style
from rich.syntax import Syntax


from textual_image.widget import Image as TextualImage

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False


def normalize_markdown(text: str) -> str:
    """
    Remove accidental leading spaces from Markdown lines,
    but preserve:
      - fenced code blocks
      - indented code blocks / nested lists
      - empty lines (paragraphs)
    """
    lines = text.splitlines()
    normalized = []
    in_fenced_code = False
    fence_char = "```"

    for line in lines:
        stripped = line.lstrip()
        
        # Toggle fenced code block
        if stripped.startswith(fence_char):
            in_fenced_code = not in_fenced_code
            normalized.append(line)  # keep as-is
            continue

        # If inside fenced code block, leave line as-is
        if in_fenced_code:
            normalized.append(line)
            continue

        # Empty line → keep
        if not stripped:
            normalized.append(line)
            continue

        # If line starts with list marker (-, *, +, 1.), keep indentation
        if re.match(r'^(\s*)([-*+]|\d+\.)\s+', line):
            normalized.append(line)
            continue

        # Otherwise: safe to remove leading spaces
        normalized.append(stripped)

    return "\n".join(normalized)

# ============================================================================
# Custom TOC Widget
# ============================================================================

class TOCEntry(Static):
    """A single TOC entry that's clickable."""
    
    def __init__(self, text: str, level: int, heading_id: str):
        super().__init__()
        self.heading_text = text
        self.level = level
        self.heading_id = heading_id
        self.indent = "  " * (level - 1)
        
    def render(self) -> Text:
        """Render the TOC entry with proper indentation."""
        rendered = Text()
        rendered.append(self.indent, style="dim")
        
        # Different bullet for different levels
        if self.level == 1:
            rendered.append("■ ", style="bold cyan")
        elif self.level == 2:
            rendered.append("▪ ", style="cyan")
        else:
            rendered.append("· ", style="dim cyan")
        
        rendered.append(self.heading_text, style="white")
        return rendered
    

class TableOfContents(VerticalScroll):
    """Custom table of contents widget."""
    
    def __init__(self):
        super().__init__(id="toc-content")
    
    async def update_headings(self, headings: list[dict]):
        """Update the TOC with new headings."""
        await self.query("*").remove()
        
        if not headings:
            await self.mount(Static("(No headings found)", classes="toc-empty"))
            return
        
        entries = []
        prev_level = 0
        
        for heading in headings:
            # Add extra spacing when going back to shallower level (e.g., h2 back to h1)
            if prev_level > 0 and heading['level'] < prev_level:
                spacer = Static("")
                spacer.styles.height = 1
                entries.append(spacer)
            
            entry = TOCEntry(
                text=heading['text'],
                level=heading['level'],
                heading_id=heading['id']
            )
            entry.add_class("toc-entry")
            entries.append(entry)
            prev_level = heading['level']
        
        await self.mount_all(entries)


# ============================================================================
# Markdown Viewer
# ============================================================================

class HeadingsExtracted(Message):
    """Message sent when headings are extracted."""
    def __init__(self, headings: list[dict]):
        super().__init__()
        self.headings = headings


class MarkdownViewer(VerticalScroll):
    """Main markdown content viewer with clean rendering and working links."""
    
    HeadingsExtracted = HeadingsExtracted

    class LoadRequest(Message):
        """Request to load a new file."""
        def __init__(self, path: Path):
            super().__init__()
            self.path = path
    
    def __init__(self):
        super().__init__(id="viewer")
        self.headings: list[dict] = []
        self.link_registry: dict[int, str | Path] = {}
        self._rendered_content_hash: int = 0

    def on_link_clicked(self, message: LinkClicked) -> None:
        """Handle link clicks."""
        target = message.target
        
        # Handle Web Links
        if isinstance(target, str) and target.startswith(('http://', 'https://')):
            self.app.notify(f"Opening: {target}", timeout=2)
            try:
                if os.name == "posix":
                    cmd = "open" if sys.platform == "darwin" else "xdg-open"
                    subprocess.Popen([cmd, target], stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
                elif os.name == "nt":
                    os.startfile(target)
            except Exception as e:
                self.app.notify(f"Failed to open URL: {e}", severity="error")
            return
        
        # Handle Local Files
        if isinstance(target, Path):
            if not target.exists():
                self.app.notify(f"File not found: {target}", severity="error")
                return
            
            suffix = target.suffix.lower()
            if suffix in ['.md', '.markdown', '.txt', '.rst']:
                self.app.notify(f"Loading: {target.name}", timeout=1)
                self.post_message(self.LoadRequest(target))
            else:
                self.app.notify(f"Opening: {target.name}", timeout=2)
                try:
                    if os.name == "posix":
                        cmd = "open" if sys.platform == "darwin" else "xdg-open"
                        subprocess.Popen([cmd, str(target)], stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
                    elif os.name == "nt":
                        os.startfile(str(target))
                except Exception as e:
                    self.app.notify(f"Failed to open: {e}", severity="error")

    async def render_markdown(self, content: str, base_path: Optional[Path] = None):
            """Render markdown content."""
            content_hash = hash(content)
            if content_hash == self._rendered_content_hash:
                return
            
            self._rendered_content_hash = content_hash
            self.base_path = base_path or Path.cwd()
            scroll_y = self.scroll_y
            
            content = normalize_markdown(content)
            
            with self.app.batch_update():
                await self.query("*").remove()
                
                # Reset registry
                self.headings = []
                self.link_registry = {}
                
                
                parser = MarkdownIt("gfm-like").enable(['table', 'strikethrough'])
                tokens = parser.parse(content)
                
                # Process markdown tokens
                widgets = await self._process_tokens(tokens, self.base_path)
                
                # Mount all markdown widgets
                await self.mount_all(widgets)
                
            
            self.scroll_y = scroll_y
            
            # Update TOC directly
            try:
                toc = self.app.query_one(TableOfContents)
                await toc.update_headings(self.headings)
            except Exception as e:
                pass
    
    async def _process_tokens(self, tokens: list[Token], base_path: Path, depth: int = 0) -> list:
            """Process markdown tokens into widgets."""
            widgets = []
            i = 0
            
            while i < len(tokens):
                token = tokens[i]
                
                if token.type == "heading_open":
                    level = int(token.tag[1])
                    i += 1
                    heading_text = ""
                    if i < len(tokens) and tokens[i].type == "inline":
                        heading_text = await self._extract_text_from_inline(tokens[i])
                    
                    # Create unique ID for this heading
                    heading_id = f"heading-{len(self.headings)}"
                    
                    # Store heading info
                    self.headings.append({
                        'level': level,
                        'text': heading_text,
                        'id': heading_id
                    })
                    
                    # Create heading widget with the same ID
                    heading_widget = Heading(level, heading_text)
                    heading_widget.id = heading_id
                    widgets.append(heading_widget)
                    i += 1
                    
                elif token.type == "table_open":
                    table_widgets = await self._process_table(tokens[i:], base_path)
                    widgets.extend(table_widgets)
                
                    while i < len(tokens) and tokens[i].type != "table_close":
                        i += 1
                    i += 1
                
                                  
                elif token.type == "paragraph_open":
                    i += 1
                    if i < len(tokens) and tokens[i].type == "inline":
                        para_widgets = await self._process_inline_to_widgets(tokens[i], base_path)
                        widgets.extend(para_widgets)
                    i += 1
                    
                elif token.type in ("fence", "code_block"):
                    code_content = token.content if token.content else ""
                    language = token.info or ""
                    if code_content.strip():
                        widgets.append(CodeBlock(code_content, language))
                    i += 1
                    
                elif token.type == "blockquote_open":
                    i += 1
                    quote_text = Text()
                    while i < len(tokens) and tokens[i].type != "blockquote_close":
                        if tokens[i].type == "inline":
                            quote_text.append(await self._process_inline(tokens[i], base_path))
                            quote_text.append("\n")
                        i += 1
                    widgets.append(BlockQuote(quote_text))
                    
                elif token.type in ("bullet_list_open", "ordered_list_open"):
                    is_ordered = token.type == "ordered_list_open"
                    start = int(token.attrGet("start") or 1) if is_ordered else 1
                    
                    i += 1
                    list_items = []
                    while i < len(tokens) and tokens[i].type != f"{'ordered' if is_ordered else 'bullet'}_list_close":
                        if tokens[i].type == "list_item_open":
                            i += 1
                            item_tokens = []
                            while i < len(tokens) and tokens[i].type != "list_item_close":
                                item_tokens.append(tokens[i])
                                i += 1
                            list_items.append(item_tokens)
                        else:
                            i += 1
                    
                    # Process the list items
                    list_widgets = await self._process_list_items(list_items, base_path, depth, 
                                                                 numbered=is_ordered, start=start)
                    widgets.extend(list_widgets)
                    
                elif token.type == "hr":
                    widgets.append(HorizontalRule())
                    i += 1
                
                else:
                    i += 1
            return widgets
    
    async def _process_list_items(self, list_items: list, base_path: Path, depth: int = 0, 
    numbered: bool = False, start: int = 1) -> list:
        """Process list items with their nested content."""
        widgets = []
        
        for idx, item_tokens in enumerate(list_items):
            item_text = Text()
            is_checkbox = False
            checked = False
            
            # First, collect all inline content from this list item
            for token in item_tokens:
                if token.type == "inline":
                    if token.children:
                        for child in token.children:
                            if child.type == "text":
                                content = child.content
                                # Check for checkbox at the beginning
                                if not item_text.plain and not is_checkbox:
                                    checkbox_match = re.match(r'^\[([ xX])\]\s*(.*)$', content)
                                    if checkbox_match:
                                        is_checkbox = True
                                        checked = checkbox_match.group(1).lower() == 'x'
                                        remaining_text = checkbox_match.group(2)
                                        if remaining_text:
                                            item_text.append(remaining_text, Style(color="#abb2bf"))
                                    else:
                                        item_text.append(content, Style(color="#abb2bf"))
                                else:
                                    item_text.append(content, Style(color="#abb2bf"))
                            elif child.type == "code_inline":
                                content = child.content
                                item_text.append(content, Style(bgcolor="#2c313a", color="#e06c75"))
                
                # Check for code blocks directly in list items
                elif token.type in ("fence", "code_block"):
                    code_content = token.content if token.content else ""
                    language = token.info or ""
                    if code_content.strip():
                        # First add the list item text if we have any
                        if item_text.plain.strip():
                            list_widget = await self._create_list_widget(item_text, depth, idx, 
                                                                        is_checkbox, checked, numbered, start)
                            widgets.append(list_widget)
                            item_text = Text()  # Reset for next content
                        
                        # Add the code block
                        code_widget = CodeBlock(code_content, language)
                        widgets.append(code_widget)
            
            # Add the list item if we have text content
            if item_text.plain.strip():
                list_widget = await self._create_list_widget(item_text, depth, idx, 
                                                            is_checkbox, checked, numbered, start)
                widgets.append(list_widget)
        
        return widgets
    
    async def _create_list_widget(self, item_text: Text, depth: int, idx: int, 
                                 is_checkbox: bool, checked: bool, 
                                 numbered: bool = False, start: int = 1) -> ListBlock:
        """Create a list widget with proper formatting."""
        prefix_text = Text()
        indent = "  " * depth
        prefix_text.append(indent, Style(color="#5c6370"))
        
        if is_checkbox:
            checkbox = "✓" if checked else "☐"
            prefix_text.append(f"{checkbox} ", Style(color="#98c379" if checked else "#5c6370"))
        elif numbered:
            prefix_text.append(f"{start + idx}. ", Style(color="#e5c07b"))
        else:
            prefix_text.append("• ", Style(color="#61afef"))
        
        final_text = prefix_text + item_text
        return ListBlock(final_text)
    
    async def _extract_text_from_inline(self, token: Token) -> str:
        if not token.children: 
            return token.content.lstrip() if token.content.startswith(' ') else token.content
        return ''.join([
            c.content.lstrip() if c.content.startswith(' ') else c.content 
            for c in token.children if c.type in ("text", "code_inline")
        ])
        
    async def _process_inline_to_widgets(self, token: Token, base_path: Path) -> list:
            """Process inline tokens into a mix of text and clickable link widgets."""
            widgets = []
            current_text = Text()
            
            if not token.children:
                return [Paragraph()] if not token.content else []
            
            style_stack = [Style(color="#abb2bf")]
            in_link = False
            link_text = ""
            link_href = ""
            
            for child in token.children:
                if child.type == "text":
                    content = child.content
                    # Only strip leading space if it's at the VERY beginning of the text node
                    # Don't strip spaces that are between words
                    if in_link:
                        link_text += content
                    else:
                        current_text.append(content, style_stack[-1])
                    
                elif child.type == "code_inline":
                    content = child.content
                    if in_link:
                        link_text += content
                    else:
                        current_text.append(content, Style(bgcolor="#2c313a", color="#e06c75"))
                    
                elif child.type == "strong_open":
                    style_stack.append(style_stack[-1] + Style(bold=True, color="#e5c07b"))
                    
                elif child.type == "em_open":
                    style_stack.append(style_stack[-1] + Style(italic=True))
                    
                elif child.type == "s_open":
                    style_stack.append(style_stack[-1] + Style(strike=True))
                    
                elif child.type == "link_open":
                    link_href = child.attrGet("href") or ""
                    link_text = ""
                    in_link = True
                    
                elif child.type == "link_close":
                    # Flush any accumulated text before the link
                    if current_text.plain:
                        para = Paragraph()
                        para.update(current_text)
                        widgets.append(para)
                        current_text = Text()
                    
                    # Determine target
                    if link_href.startswith(('http://', 'https://')):
                        target = link_href
                    else:
                        target = (base_path / link_href).resolve()
                    
                    # Create clickable link widget
                    link_id = len(self.link_registry)
                    self.link_registry[link_id] = target
                    widgets.append(ClickableLink(link_text or link_href, target, link_id))
                    
                    in_link = False
                    link_text = ""
                    link_href = ""
                    
                elif child.type == "image":
                    src = child.attrGet("src") or ""
                    alt = child.attrGet("alt") or ""
                    
                    try:
                        container = MarkdownImage(src, alt, base_path)
                        widgets.append(container)
                            
                    except Exception as e:
                        error_text = Text(f"[{alt or src} - Error: {str(e)[:30]}]", 
                                        style="color(#e06c75) italic")
                        error_widget = Paragraph()
                        error_widget.update(error_text)
                        error_widget.add_class("image-error")
                        widgets.append(error_widget)
         
                                       
                elif child.type in ("strong_close", "em_close", "s_close"):
                    if len(style_stack) > 1:
                        style_stack.pop()
                        
                elif child.type == "softbreak":
                    if current_text.plain.strip():
                                    para = Paragraph()
                                    para.update(current_text)
                                    widgets.append(para)
                                    current_text = Text()
                elif child.type == "hardbreak":
                    current_text.append("\n")
            
            # Flush any remaining text
            if current_text.plain.strip():
                para = Paragraph()
                para.update(current_text)
                widgets.append(para)
            
            return widgets if widgets else [Paragraph()]
        
    async def _process_inline(self, token: Token, base_path: Path) -> Text:
            """Process inline tokens for non-interactive contexts like blockquotes."""
            text = Text()
            style_stack = [Style(color="#abb2bf")]
            
            if not token.children:
                return text
            
            for child in token.children:
                if child.type == "text":
                    # Strip leading whitespace
                    content =  child.content
                    text.append(content, style_stack[-1])
                    
                elif child.type == "code_inline":
                    content = child.content
                    text.append(content, Style(bgcolor="#2c313a", color="#e06c75"))
                    
                elif child.type == "strong_open":
                    style_stack.append(style_stack[-1] + Style(bold=True, color="#e5c07b"))
                    
                elif child.type == "em_open":
                    style_stack.append(style_stack[-1] + Style(italic=True))
                    
                elif child.type == "s_open":
                    style_stack.append(style_stack[-1] + Style(strike=True))
                    
                elif child.type == "link_open":
                    style_stack.append(Style(color="#61afef", underline=True))
                    
                elif child.type == "link_close":
                    style_stack.pop()
                    
                elif child.type == "image":
                    pass
                    
                elif child.type in ("strong_close", "em_close", "s_close"):
                    if len(style_stack) > 1:
                        style_stack.pop()
                        
                elif child.type == "softbreak":
                    text.append(" ")
                elif child.type == "hardbreak":
                    text.append("\n")
                
            return text
    
    async def _process_table(self, tokens: list[Token], base_path: Path) -> list:
            """Render GFM tables as aligned plain text."""
            widgets = []
        
            rows: list[list[str]] = []
            current_row: list[str] = []
            in_cell = False
            cell_text = ""
        
            for token in tokens:
                if token.type in ("th_open", "td_open"):
                    in_cell = True
                    cell_text = ""
        
                elif token.type == "inline" and in_cell and token.children:
                    for child in token.children:
                        if child.type in ("text", "code_inline"):
                            cell_text += child.content
        
                elif token.type in ("th_close", "td_close"):
                    current_row.append(cell_text.strip())
                    in_cell = False
                    cell_text = ""
        
                elif token.type == "tr_close":
                    if current_row:
                        rows.append(current_row)
                        current_row = []
        
                elif token.type == "table_close":
                    break
        
            if not rows:
                return widgets
        
            # --- Compute column widths ---
            num_cols = max(len(r) for r in rows)
            EXTRA_PADDING = 2  
            
            col_widths = [
                max(len(r[i]) if i < len(r) else 0 for r in rows) + EXTRA_PADDING
                for i in range(num_cols)
            ]
            
        
            # --- Render table ---
            def render_row(row: list[str]) -> str:
                return "  ".join(
                        (row[i] if i < len(row) else "").ljust(col_widths[i])
                        for i in range(num_cols)
                    )
        
            lines = []
            lines.append(render_row(rows[0]))  # header
            lines.append(
                " ".join("-" * w for w in col_widths) 
            )
        
            for row in rows[1:]:
                lines.append(render_row(row))
        
            table_text = "\n".join(lines)
            table_text+='\n'
        
            text = Text(table_text, style=Style(color="#abb2bf"))
            widget = Static(text)
            widget.add_class("markdown-table")
            widgets.append(widget)
        
            return widgets
        
                                      
    
# ============================================================================
# Main Application
# ============================================================================

class MarkTerm(App):
    """Beautiful terminal markdown reader with interactive clickable links."""
    
    CSS = THEME
    
    BINDINGS = [
        Binding("q", "quit", "Quit", show=True),
        Binding("r", "reload", "Reload", show=True),
        Binding("t", "toggle_toc", "TOC", show=True),
        Binding("b", "back", "Back", show=True),
        Binding("n", "next_heading", "Next Heading", show=True),
    ]
    
    TITLE = "MarkTerm"
    
    current_file: reactive[Optional[Path]] = reactive(None)
    show_toc: reactive[bool] = reactive(False)
    
    def __init__(self, filepath: Optional[str] = None):
        super().__init__()
        self.filepath = Path(filepath) if filepath else None
        self._last_mtime = 0
        self._history: list[Path] = []

    def compose(self) -> ComposeResult:
        with Horizontal(id="main-container"):
            with Vertical(id="toc-panel", classes="hidden"):
                yield Static("Contents", classes="toc-title")
                yield TableOfContents()
            yield MarkdownViewer()
        yield Static("", id="status")
        yield Footer()
    
    async def on_mount(self) -> None:
        """Load initial file if provided."""
        if self.filepath:
            await self.load_file(self.filepath)
            self.set_interval(1, self.check_file_changes)
        else:
            self.sub_title = "No file loaded"
    
    async def load_file(self, filepath: Path):
        """Load and render a markdown file."""
        try:
            # Add to history
            if self.current_file and self.current_file != filepath:
                self._history.append(self.current_file)
            
            self.current_file = filepath
            
            if filepath.exists():
                self._last_mtime = filepath.stat().st_mtime
            
            content = filepath.read_text(encoding='utf-8', errors='replace')
            
            self.sub_title = str(filepath)
            
            viewer = self.query_one(MarkdownViewer)
            await viewer.render_markdown(content, filepath.parent)
            
            lines = content.count('\n') + 1
            status_widget = self.query_one("#status", Static)
            history_info = f" • {len(self._history)} back" if self._history else ""
            status_widget.update(f"{filepath.name} • {len(content):,} chars • {lines:,} lines{history_info}")
            
        except Exception as e:
            status_widget = self.query_one("#status", Static)
            status_widget.update(f"Error: {str(e)[:50]}...")
    
    async def check_file_changes(self):
        """Check if file has been modified and reload if needed."""
        if not self.current_file or not self.current_file.exists():
            return
        
        try:
            current_mtime = self.current_file.stat().st_mtime
            if current_mtime > self._last_mtime:
                # File has changed, reload it
                self._last_mtime = current_mtime
                await self.load_file(self.current_file)
        except Exception:
            pass  # Silently fail
    
    def action_reload(self):
        """Reload current file."""
        if self.current_file:
            self.notify("Reloading...", timeout=1)
            self.run_worker(self.load_file(self.current_file), exclusive=True)
    
    def action_back(self):
        """Go back to previous file."""
        if self._history:
            prev_file = self._history.pop()
            self.run_worker(self._load_without_history(prev_file), exclusive=True)
                
    async def _load_without_history(self, filepath: Path):
        """Load file without adding to history."""
        old_current = self.current_file
        self.current_file = None  # Prevent adding to history
        await self.load_file(filepath)
    
    def action_toggle_toc(self):
        """Toggle table of contents panel."""
        self.show_toc = not self.show_toc
        toc_panel = self.query_one("#toc-panel")
        toc_panel.set_class(not self.show_toc, "hidden")
        
        if self.show_toc:
            # Force TOC update when showing
            try:
                viewer = self.query_one(MarkdownViewer)
                toc = self.query_one(TableOfContents)
                self.run_worker(toc.update_headings(viewer.headings))
            except Exception as e:
                self.notify(f"TOC error: {e}", severity="error")
    
    def on_markdown_viewer_headings_extracted(self, message: HeadingsExtracted):
        """Update TOC when headings are extracted."""
        try:
            self.notify(f"Headings extracted: {len(message.headings)}", timeout=3)
            toc = self.query_one(TableOfContents)
            self.run_worker(toc.update_headings(message.headings))
        except Exception as e:
            self.notify(f"TOC error: {e}", severity="error")
    
    async def on_markdown_viewer_load_request(self, message: MarkdownViewer.LoadRequest):
        """Handle load request from viewer."""
        await self.load_file(message.path)

    def action_next_heading(self):
                    """Jump to next heading."""
                    self._navigate_headings(forward=True)
                
    
    def _navigate_headings(self, forward: bool = True):
                """Fixed - proper current heading detection."""
                try:
                    viewer = self.query_one(MarkdownViewer)
                    
                    if not viewer.headings:
                        self.notify("No headings", severity="warning")
                        return
                    
                    # Get all valid heading indices (those with widgets)
                    valid_indices = []
                    for idx, heading in enumerate(viewer.headings):
                        try:
                            widget = viewer.query_one(f"#{heading['id']}", Static)
                            if widget:
                                valid_indices.append(idx)
                        except:
                            continue
                    
                    if not valid_indices:
                        self.notify("No valid headings", severity="warning")
                        return
                    
                    
                    # Find which valid heading is CLOSEST to top of screen
                    closest_idx = valid_indices[0]
                    min_distance = float('inf')
                    
                    for idx in valid_indices:
                        try:
                            widget = viewer.query_one(f"#{viewer.headings[idx]['id']}", Static)
                            if widget:
                                distance = abs(widget.region.y - 50)
                                if distance < min_distance:
                                    min_distance = distance
                                    closest_idx = idx
                        except:
                            continue
                    
                    # Find position of closest_idx in valid_indices list
                    current_pos = valid_indices.index(closest_idx)
                    
                    # Move to next/previous in valid_indices
                    if forward:
                        if current_pos < len(valid_indices) - 1:
                            target_idx = valid_indices[current_pos + 1]
                        else:
                            target_idx = valid_indices[0]  # Wrap to first
                    
                    # Scroll
                    target_heading = viewer.headings[target_idx]
                    target_widget = viewer.query_one(f"#{target_heading['id']}", Static)
                    
                    if target_widget:
                        viewer.scroll_to_widget(target_widget, animate=True, top=True)
                        
                        text = target_heading['text'][:50] or f"Heading {target_heading['level']}"
                        direction = "Next" if forward else "Previous"
                        self.notify(f"{direction}: {text}", timeout=1)
                        
                except Exception as e:
                    self.notify(f"Error: {e}", severity="error")
        
def main():
    """Entry point."""
    if len(sys.argv) < 2:
        print("Usage: python markterm.py <file.md>")
        print("\nExample:")
        print("  python markterm.py README.md")
        print("\nControls:")
        print("  Mouse - Click on links to follow them!")
        print("  Mouse wheel or arrow keys - Scroll")
        print("  t - Toggle table of contents")
        print("  r - Reload file")
        print("  b - Go back to previous file")
        print("  q - Quit")
        sys.exit(1)
    
    filepath = sys.argv[1]
    app = MarkTerm(filepath)
    app.run()


if __name__ == '__main__':
    main()


