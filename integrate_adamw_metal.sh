#!/bin/bash
# integrate_adamw_metal.sh - Integrazione AdamW Metal Float16
set -e

cd /Volumes/Audio/masteraudio/Developer/SCMMagellano

echo "🚀 ADAMW METAL FLOAT16 - INTEGRAZIONE COMPLETA"
echo "=============================================="

# 5. BUILD TEST
echo "5/5 Build test..."
rm -rf .build

if swift build -c release 2>&1 | tee adamw_build.log; then
    echo ""
    echo "🎉 ADAMW METAL INTEGRATO!"
    echo ""
    echo "Performance attese:"
    echo "  • Speedup: 5x vs Accelerate Float32"
    echo "  • Memoria: 50% risparmio"
    echo "  • Precisione: <1e-4 errore"
    echo ""
    echo "Test:"
    echo "  .build/release/MagellanoCLI benchmark-adamw"
else
    echo "❌ Build failed - restore backup"
    cp Package.swift.backup Package.swift
    exit 1
fi
