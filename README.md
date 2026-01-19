# Markterm

**Markterm** is a modern, terminal-based **Markdown viewer** built with [Textual](https://textual.textualize.io/) and [Rich](https://github.com/Textualize/rich). It provides a beautiful, interactive way to read Markdown files directly in your terminal, including support for images, links, tables, code blocks, and more.

---

## Features

* **Beautiful Markdown Rendering** 
  Full support for headings, bold, italics, strikethrough, lists, blockquotes, and code blocks with syntax highlighting.

* **Sixel / Terminal Images**
  Display images in your terminal using Sixel graphics (if supported).

* **Tables**
  Cleanly rendered tables with proper alignment and adjustable column widths.

* **Links**
  Clickable links for both local files and web URLs. Supports navigating Markdown file links.

* **Table of Contents (TOC)**
  Automatically generates a TOC for review of headings.

* **Interactive Scrolling**
  Smooth vertical scrolling for long documents and nested lists.

* **Live**
  Any changes made to the opened document are rendered instantly giving a live preview.

---

## Installation

### From Source

```bash
git clone https://github.com/MemoritiK/Markterm.git
cd Markterm
pip install -r requirements.txt
python main.py <your-markdown-file.md>
```

### Prebuilt Binary

```bash
chmod +x Markterm
./Markterm <your-markdown-file.md>
```

---

## Usage

```bash
./Markterm README.md
```

* Navigate with **arrow keys** or **PgUp/PgDn**.
* Click links to open URLs or local files.
* Scroll through large documents smoothly.
* Supports nested lists, checkboxes, and code blocks with syntax highlighting.

---

## Screenshots
<div>
  <img src="https://github.com/user-attachments/assets/284bf845-0e00-4ba4-9395-47081a00cb63" width="59%" />
  <img src="https://github.com/user-attachments/assets/4767180a-83f3-4855-afda-ce6b3fb93fa7" width="39%" />
</div>

*(Sixel images will render inline if your terminal supports them)*

---

## Dependencies

* Python 3.10+
* [Textual](https://github.com/Textualize/textual)
* [Rich](https://github.com/Textualize/rich)
* [markdown-it-py](https://github.com/executablebooks/markdown-it-py)
* [Textual Image](https://github.com/Textualize/textual-image)


## License

MIT License – see `LICENSE` file for details.
