from textual_image.widget.sixel import Image as TextualImage
from textual.widgets import Static
from textual import events
from textual.message import Message
from pathlib import Path
from textual.containers import Vertical, Horizontal
from textual.app import ComposeResult
      
from rich.text import Text
from rich.style import Style
from rich.syntax import Syntax
from typing import Optional

class MarkdownImage(Vertical):
    """Wrapper widget for displaying markdown images."""
    
    def __init__(self, src: str, alt: str, base_path: Path):
        super().__init__()
        self.add_class("markdown-image")
        
        self.src = src
        self.alt = alt
        self.base_path = base_path
        
        if alt:
            self.tooltip = alt
    
    def compose(self) -> ComposeResult:
        """Create the image widget."""
        image_path = (self.base_path / self.src).resolve()
        
        if image_path.exists():
            try:
                # Create the actual image widget
                img_widget = TextualImage(str(image_path))
                img_widget.add_class("markdown-image")
                img_widget.styles.width = "auto"  # or "500" for pixels
                img_widget.styles.height = "auto"
                yield img_widget
                                
            except Exception as e:
                error_widget = Static(f"Failed to load: {str(e)[:30]}")
                error_widget.add_class("image-error")
                yield error_widget
        else:
            error_widget = Static(f"Not found: {self.src}")
            error_widget.add_class("image-error")
            yield error_widget
        
        # Add caption if alt text exists
        if self.alt:
            caption = Static(Text(self.alt, style="color(#56b6c2) italic"))
            caption.add_class("image-caption")
            yield caption

THEME = """
$primary: #61afef;
$secondary: #c678dd;
$accent: #56b6c2;
$warning: #e5c07b;
$success: #98c379;
$error: #e06c75;
$background: #1a1a1a;
$surface: #21252b;
$boost: #2c313a;

.links-box {
    border: cyan;
    height: auto;
}

/* Table of Contents Styling */
#toc-panel {
    width: 50;
    border-right: solid $primary;
    background: $surface;
}

#toc-panel.hidden {
    display: none;
}

.toc-title{
    padding: 1 1;
    align: center middle;
    color: #61afef;
    text-style: bold;
}

#toc-content {
    padding: 1;
    background: $surface;
}

.toc-entry {
    padding: 0 1;
    margin: 0;
    background: $surface;
}

.toc-entry:hover {
    background: $primary 20%;
    text-style: bold;
}

.toc-empty {
    color: $text-muted;
    text-align: center;
    padding: 2;
    text-style: italic;
}

Screen {
    background: $background;
}

Header {
    display: none;
}

Footer {
    background: $surface;
    color: #5c6370;
}

#main-container {
    height: 100%;
}

#viewer {
    background: $background;
    padding: 1 3;
    scrollbar-gutter: stable;
    scrollbar-size-vertical: 1;
}


/* Headings - clean, no symbols */
.h1 {
    width: 100%;
    color: #ffffff;
    text-style: bold;
    margin: 2 0 1 0;
    background: #0051a8;
    padding: 1 2;
    text-align: center;
}

.h2 {
    color: $secondary;
    text-style: bold;
    margin: 1 0 1 0;
    padding: 0 1;
    text-align: center;
}

.h3 {
    color: $accent;
    text-style: bold;
    margin: 1 0;
    border-bottom: solid #3e4451;
}

.h4 {
    color: $warning;
    text-style: bold;
    margin: 1 0;
}

.h5 {
    color: $success;
    text-style: bold;
    margin: 1 0;
}

.h6 {
    color: #abb2bf;
    text-style: bold;
    margin: 1 0;
}

/* Content blocks */
.paragraph {
    color: #abb2bf;
    margin: 0 0 1 0;
}

.code-block {
    background: #2c313a;
    margin: 1 0;
    padding: 1 2;
}

.blockquote {
    border-left: thick $success;
    padding-left: 2;
    margin: 1 0;
    color: #abb2bf;
}

.list-block {
    margin: 0 0 1 0;
    color: #abb2bf;
}

.hr-block {
    height: 0;          /* No height needed */
    margin: 2 0;
    border-top: tall #0051a8;  /* 'tall' = 1px thin line */
}

#status {
    dock: bottom;
    height: 1;
    background: $surface;
    color: #5c6370;
    padding: 0 2;
}

ClickableLink {
    width: auto;
    height: auto;
}

ClickableLink:hover {
    text-style: bold;
}
.markdown-image {
    height: 20;}
    
.markdown-table{
    align: center middle;
    text-align: center;
    
}

"""


class MarkdownBlock(Static):
    """Base class for markdown blocks."""
    pass


class ClickableLink(Static):
    """A clickable link widget."""
    
    def __init__(self, text: str, target: str | Path, link_id: int):
        super().__init__()
        self.border_title = None
        self.target = target
        self.link_id = link_id
        
        # Style the link text
        if isinstance(target, str) and target.startswith(('http://', 'https://')):
            link_text = Text(text, style=Style(color="#61afef", underline=True))
        else:
            link_text = Text(text, style=Style(color="#98c379", underline=True))
        
        self.update(link_text)
    
    def on_click(self, event: events.Click) -> None:
        """Handle click on link."""
        event.stop()
        self.post_message(LinkClicked(self.target, self.link_id))


class LinkClicked(Message):
    """Message posted when a link is clicked."""
    def __init__(self, target: str | Path, link_id: int):
        super().__init__()
        self.target = target
        self.link_id = link_id


class Heading(MarkdownBlock):
    """Heading widget - NO markdown symbols displayed."""
    
    def __init__(self, level: int, text: str):
        super().__init__()
        self.add_class(f"h{level}")
        self.update(text)


class Paragraph(MarkdownBlock):
    """Paragraph block that can contain inline widgets."""
    
    def __init__(self):
        super().__init__()
        self.add_class("paragraph")


class CodeBlock(MarkdownBlock):
    """Code block with syntax highlighting - NO line numbers, NO glow."""
    
    def __init__(self, code: str, language: str):
        super().__init__()
        self.add_class("code-block")
        
        try:
            syntax = Syntax(
                code.rstrip(),
                language or "text",
                theme="monokai",
                line_numbers=False,
                word_wrap=False,
                background_color="#2c313a",
                padding=0,
            )
            self.update(syntax)
        except Exception:
            self.update(Text(code, style="color(#abb2bf) bgcolor(#2c313a)"))


class BlockQuote(MarkdownBlock):
    """Blockquote block - NO > symbols shown."""
    
    def __init__(self, content: Text):
        super().__init__()
        self.add_class("blockquote")
        self.update(content)


class ListBlock(MarkdownBlock):
    """List block."""
    
    def __init__(self, content: Text):
        super().__init__()
        self.add_class("list-block")
        self.update(content)


class HorizontalRule(MarkdownBlock):
    """Horizontal rule."""
    
    def __init__(self):
        super().__init__()
        self.add_class("hr-block")
        self.update("")

