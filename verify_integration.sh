#!/bin/bash
# Pre-Integration Verification Script

cd ~/Developer/SCMMagellano

echo "🔍 Pre-Integration Verification"
echo ""

# 1. Check existing structure
echo "📁 Checking directory structure..."
mkdir -p Sources/MagellanoCore/Data
mkdir -p Sources/MagellanoCore/Training

# 2. Verify critical files exist
echo "🔎 Checking dependencies..."

FILES_TO_CHECK=(
    "Sources/MagellanoCore/Training/LoRALayer.swift"
    "Sources/MagellanoCore/Quantization/ModelQuantizer.swift"
    "Sources/MagellanoCore/Core/MemoryProfiler.swift"
    "Sources/MagellanoCore/Logging/StructuredLogger.swift"
)

MISSING=0
for FILE in "${FILES_TO_CHECK[@]}"; do
    if [ -f "$FILE" ]; then
        echo "  ✅ $FILE"
    else
        echo "  ❌ MISSING: $FILE"
        MISSING=$((MISSING + 1))
    fi
done

if [ $MISSING -gt 0 ]; then
    echo ""
    echo "⚠️  Missing $MISSING critical files!"
    echo "   Integration may fail. Review files first."
    exit 1
fi

echo ""
echo "✅ All dependencies present"
echo ""

# 3. Check LoRALayer API
echo "🔎 Checking LoRALayer API..."
if grep -q "matrixA" Sources/MagellanoCore/Training/LoRALayer.swift; then
    echo "  ✅ matrixA found"
else
    echo "  ❌ matrixA not found in LoRALayer"
fi

if grep -q "matrixB" Sources/MagellanoCore/Training/LoRALayer.swift; then
    echo "  ✅ matrixB found"
else
    echo "  ❌ matrixB not found in LoRALayer"
fi

echo ""
echo "📋 Next Steps:"
echo "1. Copy DataLoader.swift to Sources/MagellanoCore/Data/"
echo "2. Copy Loss.swift, Optimizer.swift, TrainingLoop.swift to Sources/MagellanoCore/Training/"
echo "3. Run: swift build -c release"
echo "4. Fix any compilation errors"
echo "5. Create unit test for each component"
echo ""
echo "Ready to proceed? (y/n)"
