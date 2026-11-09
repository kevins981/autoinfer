#!/bin/bash
# nvcc wrapper for torch's JIT extension builds (cpp_extension.load / load_inline), set as
# PYTORCH_NVCC by env.sh. Makes those builds hit the shared ccache across workspaces.
#
# Why: torch builds each JIT extension in $TORCH_EXTENSIONS_DIR/<name>/ with the source given
# as a path relative to that folder (../../<workspace>/.../x.cu). A new workspace (every
# official eval creates one; workers create many) means a new path, so ninja recompiles the
# extension (~40 s each, inside the GPU job, both TP ranks waiting) even when the source is
# byte-identical to one compiled before. ccache alone doesn't help: the path is in the
# preprocessed output. (test-6: official trial-1 starts took 160-210 s vs 60-80 s warm.)
#
# What: each source file argument is copied to a content-addressed path
# ($AUTOINFER_JIT_SRC/<sha1>/<basename>, one copy per content) and nvcc is run on that copy
# through ccache, so identical sources compile to identical ccache keys from any workspace.
# Sources with local includes (#include "...") are passed through unchanged (the copy would
# break them); they just don't get cached across paths. Without ccache: plain nvcc.
# ninja still tracks the workspace file (it is the build's input), so edits rebuild as before.
set -u
real_nvcc=${AUTOINFER_REAL_NVCC:-${CUDA_HOME:+$CUDA_HOME/bin/nvcc}}
real_nvcc=${real_nvcc:-nvcc}
if ! command -v ccache >/dev/null 2>&1 || [ -z "${AUTOINFER_JIT_SRC:-}" ]; then
  exec "$real_nvcc" "$@"
fi
args=()
for a in "$@"; do
  case "$a" in
    -*) ;;
    *.cu|*.cpp|*.cc|*.cxx|*.c)
      if [ -f "$a" ] && ! grep -q '^[[:space:]]*#[[:space:]]*include[[:space:]]*"' "$a"; then
        h=$(sha1sum "$a" | cut -c1-16)
        d=$AUTOINFER_JIT_SRC/$h
        s=$d/$(basename "$a")
        if [ ! -f "$s" ]; then
          mkdir -p "$d"
          cp -p "$a" "$s.tmp.$$" && mv -f "$s.tmp.$$" "$s"     # atomic: two ranks may race
        fi
        a=$s
      fi ;;
  esac
  args+=("$a")
done
exec ccache "$real_nvcc" "${args[@]}"
