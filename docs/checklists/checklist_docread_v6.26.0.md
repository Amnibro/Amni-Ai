# Checklist: DOCREAD (v6.26.0) — document reading and coding harnesses

Goal: Adam reads and answers from PDFs, images and other files through a harness (lane H), and does agentic coding by harnessing Amni-Code in a sandbox. The document harness extracts structured text with locators, retrieves the passages that answer the question, and hands them to the blend core as grounded facts with citations. The coding harness copies the project into a temporary sandbox, lets Amni-Code (Adam as its model) work and validate there, places the validated changes into the target and deletes the sandbox.

## Tool (tools/docread, Helium via Gold)
- [x] dm: document model, Markdown/JSON output, chunker with heading paths and locators, BM25 ask
- [x] dcrypt: MD5, RC4, AES inverse cipher and CBC, Standard security handler R2-R6 (vectors pass); damaged O/U/OE/UE lengths fail cleanly instead of crashing
- [x] pdf: lexer, objects, xref tables and streams, hybrid files, object streams, rebuild fallback, filters, encryption hookup, page tree, metadata, outline, form fields, annotations
- [x] glyph: Adobe glyph list subset, StandardEncoding, base-14 widths
- [x] pfont: simple and composite fonts, ToUnicode, embedded encodings, Type1 built-in encodings, TrueType cmap fallback, widths
- [x] pcontent: content interpreter, form XObjects, ActualText, inline images, scanned-page detection
- [x] playout: spans with words, word-gap column tables, blocks, reading order, paragraphs, lists, headings, running header and footer removal, ligature expansion, squeezed-space joins
- [x] zip, xml, office (DOCX, XLSX, PPTX, ODT, ODS, ODP)
- [x] html, rtf, textfmt (text, Markdown, CSV, TSV, JSON), eml (MIME, attachments; sniffed before markup)
- [x] image: PNG, JPEG, GIF, WebP, TIFF, BMP, ICO, HEIC, AVIF, SVG metadata, EXIF/XMP/IPTC text, SVG text (19 fixtures vs magick identify)
- [x] pimage: PDF image extraction for OCR (paint order, shared resources, predictors, RC4/AES), pixel-identical to pdfimages on the supported color spaces
- [x] pack: gzip, bzip2, tar with caps; extra: Jupyter notebooks, iCalendar, vCard, source code split by definitions
- [x] ole: legacy .doc/.xls/.ppt/.msg through the cfb container reader; .xls sheets and .ppt slides carry their number as the locator (units "sheet" and "slide")
- [x] ofcrypt: password-protected Office files (ECMA-376 Agile with SHA-1/256/384/512 and AES-128/192/256, Standard with AES-128/192/256); RC4 CryptoAPI and extensible encryption are refused with a warning; 11 fixtures checked against msoffcrypto-tool
- [x] Passwords: `DOCREAD_PASSWORD` opens PDF (user or owner password, R2-R6) and Office files; the empty password and "VelvetSweatshop" are tried too; a decrypted Office file reports its real format (docx, xlsx, pptx) with meta encrypted=yes
- [x] mbox: a mailbox splits into "Message N" sections (first 1000 messages), each read as an email with its attachments
- [x] xlsx sheets carry their number as the locator, matching .xls
- [x] lzma and sevenz: .xz (CRC32/CRC64/SHA-256, multi-stream, BCJ x86 and Delta filters), legacy .lzma, and 7z (LZMA, LZMA2, Deflate, BZip2, Copy, BCJ, Delta, solid folders, encoded headers); AES, PPMd, BCJ2 and ARM64 are refused with a warning; 41 fixtures checked against xz and 7z, including truncation and header fuzz sweeps
- [x] docread main: sniffing, EPUB, generic XML, ZIP/TAR/GZIP/BZIP2 recursion, md/json/info/chunks/ask/sniff/images, MCP server mode
- [x] Tests: crypto vectors, office/text/image/archive fixtures with expected output, PDF fixtures (LibreOffice, RC4-40/128, AES-128, AES-256 R6, object streams, PDF 2.0, scanned images), oracle word F1 vs pdftotext on 14 public PDFs

## Adam integration
- [x] Backups in backups/ before editing ingest.py, amni_serve.py, jarvis_web.py, code_harness.py, Gold specs
- [x] amni/serve/docread.py: binary lookup, run with timeout, session document store, OCR sidecars for images and scanned PDF pages, captions through the vision path when the model is cached
- [x] ingest_file skill (all formats, pypdf fallback kept for ingest_pdf), read_document skill (ask passages)
- [x] doc endpoints: attach (base64 or allowed path), list, ask, clear
- [x] attach takes an optional "password": a protected upload without it is refused with a clear error; with it only the decrypted text is kept as a derived sidecar and citations come from it (via "password"); the password goes to docread through one child process's environment and is never stored or logged
- [x] Real OCR test with tesseract (an invented rendered note PNG and a scanned PDF of it); skipped when tesseract or its language data is missing
- [x] /chat/stream: documents attached to the session → top passages with citations into user_facts; skip the LUT shortcut when grounded
- [x] jarvis_web: attach button and drag-and-drop; /code timeline with sandbox, validation and placement
- [x] code_harness: per-run sandbox copy (reflink when possible), Amni-Code in bubblewrap with the target read-only, docread MCP server seeded, validation in the copy, all-or-nothing placement with conflict detection, new cache/build files never placed, patch kept, sandbox deleted
- [x] Amni-Code: lenient text tool-call JSON for small local models (branch fix/lenient-tool-json, v3.0.1)
- [x] Lint, tests, live end-to-end coding run (granite4.2 through Ollama: tests failing → fixed in the sandbox → validated → placed → sandbox deleted)
- [ ] architecture_map.md and changelog.md entries
- [ ] Owner confirmation
