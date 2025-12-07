# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This repository contains classic horror literature texts from Project Gutenberg. It serves as a collection of public domain horror novels for text analysis, natural language processing, or literary study.

## Contents

The repository includes the following classic horror texts:

- **Carmilla.txt** (3,721 lines) - Joseph Sheridan Le Fanu's vampire novella
- **Dracula.txt** (15,849 lines) - Bram Stoker's classic vampire novel
- **Frankenstein.txt** (7,737 lines) - Mary Shelley's gothic science fiction novel
- **The Strange Case of Dr. Jekyll and Mr. Hyde.txt** (2,930 lines) - Robert Louis Stevenson's psychological horror novella
- **Turning of the Screw.txt** (4,926 lines) - Henry James's ghost story

## File Format

All text files are:
- UTF-8 encoded with BOM (Byte Order Mark)
- Use CRLF (Windows-style) line terminators
- Include Project Gutenberg license headers and metadata at the beginning

## Working with the Texts

When processing these files:

1. **Encoding**: Files use UTF-8 with BOM. When reading programmatically, handle the BOM appropriately:
   - Python: Use `encoding='utf-8-sig'` to automatically strip the BOM
   - Other languages: Skip the first 3 bytes (0xEF 0xBB 0xBF) or use BOM-aware readers

2. **Line Endings**: Files use CRLF (`\r\n`). Most modern text processing tools handle this automatically, but be aware when doing byte-level operations.

3. **Content Structure**: Each file begins with Project Gutenberg metadata (title, author, release date, license) followed by "START OF THE PROJECT GUTENBERG EBOOK" marker before the actual literary content begins.

## Source

All texts are from Project Gutenberg (www.gutenberg.org) and are in the public domain in the United States and most other countries.
