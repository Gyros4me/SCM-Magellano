#!/bin/bash
# assemble_and_build.sh - Assembla progetto e fa primo build test

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo "============================================================"
echo "  🏗️  MAGELLANO TAHOE - Assemble & First Build"
echo "============================================================"
echo ""

# Configuration
PROJECT_DIR="${1:-$HOME/Development/MagellanoTahoe}"
FILES_DIR="."

echo "📋 Configuration:"
echo "   Project directory: $PROJECT_DIR"
echo "   Source files: $FILES_DIR"
echo ""

# Check if FILES_DIR exists
if [ ! -d "$FILES_DIR" ]; then
    echo -e "${RED}❌ Error: $FILES_DIR not found${NC}"
    echo "Make sure you're running this from the correct context"
    exit 1
fi

# 1. Create project structure if not exists
echo -e "${BLUE}1/8${NC} Checking project structure..."
if [ ! -d "$PROJECT_DIR" ]; then
    echo "   Creating project directory..."
    mkdir -p "$PROJECT_DIR"
fi
cd "$PROJECT_DIR"
echo -e "${GREEN}✓${NC} Project directory ready"

# 2. Copy Package.swift
echo -e "${BLUE}2/8${NC} Installing Package.swift..."
if [ -f "$FILES_DIR/Package.swift" ]; then
    cp "$FILES_DIR/Package.swift" ./Package.swift
    echo -e "${GREEN}✓${NC} Package.swift installed"
else
    echo -e "${RED}❌ Package.swift not found${NC}"
    exit 1
fi

# 3. Create minimal source structure
echo -e "${BLUE}3/8${NC} Creating source directories..."
mkdir -p Sources/{MagellanoCore,MagellanoMetal/Kernels,MagellanoTraining,MagellanoData,MagellanoPrivacy,MagellanoCheckpoint,MagellanoCLI}
mkdir -p Tests/{MagellanoCoreTests,MagellanoMetalTests,MagellanoTrainingTests,MagellanoPrivacyTests}
mkdir -p {Documentation,Configs,Scripts,Resources}
echo -e "${GREEN}✓${NC} Directory structure created"

# 4. Install core files
echo -e "${BLUE}4/8${NC} Installing source files..."

# MagellanoCore/Tensor.swift
if [ -f "$FILES_DIR/Tensor.swift" ]; then
    cp "$FILES_DIR/Tensor.swift" Sources/MagellanoCore/Tensor.swift
    echo -e "${GREEN}✓${NC} Tensor.swift installed"
fi

# MagellanoCLI/main.swift
if [ -f "$FILES_DIR/main.swift" ]; then
    cp "$FILES_DIR/main.swift" Sources/MagellanoCLI/main.swift
    echo -e "${GREEN}✓${NC} main.swift installed"
fi

# 5. Create placeholder files for other modules
echo -e "${BLUE}5/8${NC} Creating placeholder files..."

# MagellanoMetal placeholder
cat > Sources/MagellanoMetal/MetalDevice.swift << 'EOF'
// Sources/MagellanoMetal/MetalDevice.swift
// Metal GPU device management

import Metal
import Foundation

public enum MetalDevice {
    public static func getDefaultDevice() throws -> MTLDevice {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MetalError.noDeviceFound
        }
        return device
    }
}

public enum MetalError: Error {
    case noDeviceFound
    case compilationFailed
}
EOF

# MagellanoTraining placeholder
cat > Sources/MagellanoTraining/Optimizer.swift << 'EOF'
// Sources/MagellanoTraining/Optimizer.swift
// Optimizer protocol

import Foundation
import MagellanoCore

public protocol Optimizer {
    func step(parameters: inout [Tensor], gradients: [Tensor])
    func zeroGradients()
}
EOF

# MagellanoData placeholder
cat > Sources/MagellanoData/DataLoader.swift << 'EOF'
// Sources/MagellanoData/DataLoader.swift
// Data loading utilities

import Foundation
import MagellanoCore

public struct DataBatch {
    public let inputs: Tensor
    public let targets: Tensor
    
    public init(inputs: Tensor, targets: Tensor) {
        self.inputs = inputs
        self.targets = targets
    }
}

public struct DataLoader {
    public init() {}
}
EOF

# MagellanoPrivacy placeholder
cat > Sources/MagellanoPrivacy/PrivacyEngine.swift << 'EOF'
// Sources/MagellanoPrivacy/PrivacyEngine.swift
// Privacy and GDPR compliance

import Foundation

public struct PrivacyEngine {
    public init() {}
}
EOF

# MagellanoCheckpoint placeholder
cat > Sources/MagellanoCheckpoint/CheckpointManager.swift << 'EOF'
// Sources/MagellanoCheckpoint/CheckpointManager.swift
// Checkpoint management

import Foundation
import MagellanoCore

public struct Checkpoint {
    public let epoch: Int
    public let modelState: [String: Any]
    
    public init(epoch: Int, modelState: [String: Any]) {
        self.epoch = epoch
        self.modelState = modelState
    }
}

public struct CheckpointManager {
    public init() {}
}
EOF

echo -e "${GREEN}✓${NC} Placeholder files created"

# 6. Create .gitignore
echo -e "${BLUE}6/8${NC} Creating .gitignore..."
cat > .gitignore << 'EOF'
.build/
Packages/
*.xcodeproj
xcuserdata/
.DS_Store
*.ckpt
*.log
.env
EOF
echo -e "${GREEN}✓${NC} .gitignore created"

# 7. Export SDK_VERSION for build
echo -e "${BLUE}7/8${NC} Setting up build environment..."
export SDK_VERSION=$(xcrun --sdk macosx --show-sdk-version)
echo "   SDK_VERSION=$SDK_VERSION"
echo -e "${GREEN}✓${NC} Environment configured"

# 8. First build test!
echo -e "${BLUE}8/8${NC} Running first build test..."
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Building MagellanoTahoe..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Clean first
swift package clean

# Resolve dependencies
echo "📦 Resolving dependencies..."
swift package resolve

# Build
echo "🔨 Building..."
swift build -c release

# Check build result
if [ $? -eq 0 ]; then
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo -e "${GREEN}✅ BUILD SUCCESSFUL!${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    
    # Test the executable
    echo "🧪 Testing executable..."
    .build/release/magellano-train info
    
    echo ""
    echo "============================================================"
    echo -e "${GREEN}🎉 PROJECT READY!${NC}"
    echo "============================================================"
    echo ""
    echo "📁 Project location: $PROJECT_DIR"
    echo "🔧 Executable: .build/release/magellano-train"
    echo ""
    echo "Next steps:"
    echo "  1. cd $PROJECT_DIR"
    echo "  2. .build/release/magellano-train info"
    echo "  3. .build/release/magellano-train train --help"
    echo ""
else
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo -e "${RED}❌ BUILD FAILED${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "Check the error messages above for details."
    exit 1
fi
