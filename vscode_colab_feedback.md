# User Feedback: VS Code Google Colab Extension
**Date:** December 26, 2025  
**Topic:** Usability Frictions & Compatibility Issues (Mech Interp Workflow)

## Executive Summary
While the ability to connect to a Google Colab backend directly from VS Code is a game-changer for utilizing free GPU compute while maintaining a local IDE workflow, several friction points significantly hinder productivity. The most critical issues involve **interactive visualization rendering (Plotly)**, **remote file system abstraction leaks**, and **secrets management**.

## Detailed Issues

### 1. Interactive Visualization Failure (Critical)
The most blocking issue encountered is the inability to render interactive Plotly figures within the VS Code notebook interface when connected to Colab.

*   **Error:** `Error rendering output item using 'jupyter-ipywidget-renderer': i is not a function`
*   **Context:** Using standard `plotly.graph_objects` or `px`.
*   **Impact:** Forces users to resort to static images (`renderer='png'`) or downloading HTML artifacts manually, breaking the exploratory data analysis loop central to scientific workflows.
*   **Attempted Fixes:**
    *   Changing renderers (`pio.renderers.default = 'colab'`, `'iframe'`, `'plotly_mimetype'`) failed.
    *   Installing `ipywidgets` specific versions failed to resolve the renderer mismatch.

### 2. File System & Path Resolution Confusion
There is a confusing disconnect between the "Local" file view in VS Code and the "Remote" execution environment in Colab.

*   **Issue:** `FileNotFoundError` for relative paths.
*   **Context:**
    *   VS Code Workspace: `C:\Users\HomePC\Project` (Windows)
    *   Colab Kernel CWD: `/content` (Linux)
*   **Friction:**
    *   User expects `Path('../../config')` to work relative to the notebook file they are editing locally.
    *   **Reality:** The notebook is executing in a remote container where those files don't exist unless explicitly cloned/uploaded.
    *   **Suggestion:** Better visual indicators of "Remote CWD" vs "Local Workspace" or auto-mounting suggestions.

### 3. Secrets & Environment Management
Managing sensitive tokens (HF_TOKEN) is cumbersome compared to local `python-dotenv`.

*   **Issue:** `.env` files are correctly gitignored locally but thus don't exist in the remote Colab environment.
*   **Friction:**
    *   VS Code's "Notebook Access" to local `.env` doesn't automatically propagate to the remote Colab kernel environment variables.
    *   Users must manually use `google.colab.userdata` (which isn't standard Python) or manually paste tokens, breaking code portability between Local/Colab.

### 4. Module Synchronization (The "Git Pull" Trap)
When developing custom modules (`src/`) alongside notebooks:

*   **Issue:** Pushing code fixes to GitHub and running `!git pull` in the notebook cell updates the *files*, but **not the loaded Python modules** in memory.
*   **Friction:**
    *   Requires a full **Restart Kernel**, which is slow and clears all cached activiations/models (expensive in ML).
    *   `importlib.reload` is flaky for deep dependencies.
*   **Suggestion:** A "Sync & Reload" feature in the extension that handles file synchronization + smart module reloading would be a killer feature.

### 5. OOM & Resource Visibility
*   **Issue:** `CUDA out of memory` errors appear without warning.
*   **Friction:** The VS Code UI lacks the resource usage bars (RAM/VRAM/Disk) that the native Colab web UI provides. Users are flying blind regarding GPU memory pressure until it crashes.

## Conclusion
The extension is promising but feels like "SSH-ing into a remote server" rather than a seamless "Remote Kernel" experience. The visualization bugs are the primary blocker for adoption in data science workflows.

**Recommendation:** Prioritize fixing the `jupyter-ipywidget-renderer` compatibility for remote Colab connections.
