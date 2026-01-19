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
