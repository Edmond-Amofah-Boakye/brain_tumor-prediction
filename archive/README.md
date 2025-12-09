# Archive Folder

This folder contains original/legacy files that have been replaced by the refactored application.

## Files

### `app_original.py`
- **Original:** Single-file Streamlit application (700+ lines)
- **Replaced by:** `app/` folder structure with modular components
- **Date archived:** December 1, 2025
- **Reason:** Code refactoring - split into services, components, and pages for better maintainability
- **Status:** WORKING BACKUP - can still be run if needed

## Running the Original App (if needed)

```bash
streamlit run archive/app_original.py
```

## Why Archived?

The original `app.py` was refactored into a professional, modular structure:

**Old Structure:**
- Single file: `app.py` (700+ lines)
- All functionality mixed together
- Hard to maintain and extend

**New Structure:**
- `app/main.py` - Main application
- `app/services/` - Business logic (model, symmetry, explanations, reports)
- `app/components/` - UI components (charts, visualizations)
- `app/pages/` - Multi-page app (Data Explorer)
- `app/config.py` - Centralized configuration

## Deleted Files

The following files were removed during cleanup:

1. **`logger.py`** - Empty file (no content)
2. **`.env`** - Empty file (no content)
3. **`utils/`** - Empty folder (no files)

## Current Application

To run the new refactored application:

```bash
streamlit run app/main.py
```

Features:
- ✅ Enhanced symmetry visualizations (radar chart)
- ✅ Clinical threshold indicators
- ✅ Data exploration page
- ✅ Modular, maintainable code
- ✅ Multi-page interface

---

**Note:** This archive is kept for reference and as a backup. The new application (`app/main.py`) is recommended for all use.
