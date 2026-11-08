#!/bin/bash
set -euo pipefail
resources="$(cd "$(dirname "$0")/../Resources" && pwd)"
cache="$HOME/Library/Application Support/Batch Deconvolution"
mkdir -p "$cache"
exec >>"$cache/launcher.log" 2>&1
runtime="$cache/runtime-@BUILD_ID@"
unset PYTHONHOME PYTHONPATH
unset JAVA_TOOL_OPTIONS JDK_JAVA_OPTIONS _JAVA_OPTIONS CLASSPATH
export PATH="$runtime/bin:$PATH"
if [ ! -f "$runtime/.ready" ]; then
    # Never run two relocations against the same environment.
    mkdir "$runtime.lock" || exit 1
    trap 'rmdir "$runtime.lock"' EXIT
    mkdir -p "$runtime"
    tar -xzf "$resources/runtime.tar.gz" -C "$runtime"
    "$runtime/bin/python" "$runtime/bin/conda-unpack"
    "$runtime/@JAVA_HOME@/bin/java" -version
    touch "$runtime/.ready"
    rmdir "$runtime.lock"
    trap - EXIT
fi
export CONDA_PREFIX="$runtime"
export PATH="$runtime/bin:$PATH"
export JAVA_HOME="$runtime/@JAVA_HOME@"
export BFF_JAVA_VERSION=21 BFF_JAVA_FETCH=never BIOFORMATS_VERSION=8.5.0
export BATCH_DECONVOLUTION_WORKSPACE="$HOME/Documents/Batch Deconvolution"
cd "$resources/app"
exec "$runtime/bin/python" -u main.py "$@"
