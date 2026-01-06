#!/bin/bash
# verify_environment.sh - Verifica ambiente Magellano Tahoe 26.2

set -e

echo "============================================================"
echo "  🔍 MAGELLANO - Environment Verification"
echo "============================================================"
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 1. macOS Version
echo "1️⃣  Checking macOS version..."
MACOS_VERSION=$(sw_vers -productVersion)
echo "   macOS: $MACOS_VERSION"

if [[ "$MACOS_VERSION" == 26.2* ]] || [[ "$MACOS_VERSION" > "26.2" ]]; then
    echo -e "   ${GREEN}✅ macOS Tahoe 26.2+ detected${NC}"
else
    echo -e "   ${RED}❌ macOS 26.2+ required, found $MACOS_VERSION${NC}"
    exit 1
fi

# 2. Xcode Version
echo ""
echo "2️⃣  Checking Xcode version..."
if command -v xcodebuild &> /dev/null; then
    XCODE_VERSION=$(xcodebuild -version | head -n1 | awk '{print $2}')
    echo "   Xcode: $XCODE_VERSION"
    
    if [[ "$XCODE_VERSION" == 26.2* ]] || [[ "$XCODE_VERSION" > "26.2" ]]; then
        echo -e "   ${GREEN}✅ Xcode 26.2+ detected${NC}"
    else
        echo -e "   ${YELLOW}⚠️  Xcode 26.2 recommended, found $XCODE_VERSION${NC}"
    fi
else
    echo -e "   ${RED}❌ Xcode not found${NC}"
    exit 1
fi

# 3. Swift Version
echo ""
echo "3️⃣  Checking Swift version..."
SWIFT_VERSION=$(swift --version | head -n1 | awk '{print $4}')
echo "   Swift: $SWIFT_VERSION"

if [[ "$SWIFT_VERSION" == 6.2* ]]; then
    echo -e "   ${GREEN}✅ Swift 6.2.x detected${NC}"
else
    echo -e "   ${YELLOW}⚠️  Swift 6.2.x recommended, found $SWIFT_VERSION${NC}"
fi

# 4. Swift Compiler Mode
echo ""
echo "4️⃣  Checking Swift compiler features..."
swift --version | grep -q "Swift version 6" && echo -e "   ${GREEN}✅ Swift 6 language mode available${NC}"

# 5. Metal Support
echo ""
echo "5️⃣  Checking Metal support..."
if system_profiler SPDisplaysDataType 2>/dev/null | grep -q "Metal"; then
    METAL_VERSION=$(system_profiler SPDisplaysDataType | grep "Metal" | head -n1 | awk '{print $3}')
    echo "   Metal: Supported (Family $METAL_VERSION)"
    echo -e "   ${GREEN}✅ Metal GPU acceleration available${NC}"
else
    echo -e "   ${YELLOW}⚠️  Metal support detection inconclusive${NC}"
fi

# 6. Hardware Info
echo ""
echo "6️⃣  Hardware information..."
CPU_BRAND=$(sysctl -n machdep.cpu.brand_string)
echo "   CPU: $CPU_BRAND"

RAM_GB=$(sysctl -n hw.memsize | awk '{print $1/1024/1024/1024}')
echo "   RAM: ${RAM_GB} GB"

if [[ "$CPU_BRAND" == *"M4"* ]] || [[ "$CPU_BRAND" == *"M5"* ]]; then
    echo -e "   ${GREEN}✅ Apple Silicon M4/M5 detected${NC}"
    
    # Check unified memory
    if (( $(echo "$RAM_GB >= 16" | bc -l) )); then
        echo -e "   ${GREEN}✅ 16GB+ unified memory available${NC}"
    else
        echo -e "   ${YELLOW}⚠️  16GB+ recommended, found ${RAM_GB}GB${NC}"
    fi
else
    echo -e "   ${YELLOW}⚠️  M4+ recommended for optimal performance${NC}"
fi

# 7. Storage Check
echo ""
echo "7️⃣  Checking storage..."
df -h / | tail -1 | awk '{print "   Internal SSD: " $4 " available (" $5 " used)"}'

# Check for external SSD (common mount points)
if [ -d "/Volumes/SSD" ]; then
    df -h /Volumes/SSD | tail -1 | awk '{print "   External SSD: " $4 " available (" $5 " used)"}'
    echo -e "   ${GREEN}✅ External SSD mounted at /Volumes/SSD${NC}"
elif [ -d "/Volumes/ACASIS" ]; then
    df -h /Volumes/ACASIS | tail -1 | awk '{print "   External SSD: " $4 " available (" $5 " used)"}'
    echo -e "   ${GREEN}✅ External SSD mounted at /Volumes/ACASIS${NC}"
elif [ -d "/Volumes/Audio" ]; then
    df -h /Volumes/Audio | tail -1 | awk '{print "   External SSD: " $4 " available (" $5 " used)"}'
    echo -e "   ${GREEN}✅ External SSD mounted at /Volumes/Audio${NC}"
else
    echo -e "   ${YELLOW}⚠️  No external SSD detected${NC}"
    echo "   💡 You'll need 2TB external SSD for datasets/checkpoints"
fi

# 8. SDK Version
echo ""
echo "8️⃣  Checking SDK version..."
SDK_PATH=$(xcrun --sdk macosx --show-sdk-path)
SDK_VERSION=$(xcrun --sdk macosx --show-sdk-version)
echo "   SDK Path: $SDK_PATH"
echo "   SDK Version: $SDK_VERSION"

if [[ "$SDK_VERSION" == 26.2* ]] || [[ "$SDK_VERSION" > "26.2" ]]; then
    echo -e "   ${GREEN}✅ macOS 26.2+ SDK available${NC}"
    export SDK_VERSION="26.2"
else
    echo -e "   ${YELLOW}⚠️  SDK version $SDK_VERSION detected${NC}"
fi

# 9. Developer Tools
echo ""
echo "9️⃣  Checking developer tools..."

# Check git
if command -v git &> /dev/null; then
    GIT_VERSION=$(git --version | awk '{print $3}')
    echo -e "   ${GREEN}✅ git $GIT_VERSION${NC}"
else
    echo -e "   ${YELLOW}⚠️  git not found${NC}"
fi

# Check xcrun
if command -v xcrun &> /dev/null; then
    echo -e "   ${GREEN}✅ xcrun available${NC}"
else
    echo -e "   ${RED}❌ xcrun not available${NC}"
fi

# 10. Network (for offline verification later)
echo ""
echo "🔟 Network status..."
if ping -c 1 github.com &> /dev/null; then
    echo -e "   ${GREEN}✅ Internet connection available${NC}"
    echo "   💡 Good for downloading dependencies now"
else
    echo -e "   ${YELLOW}⚠️  No internet connection${NC}"
    echo "   💡 Will need offline mode with pre-cached models"
fi

echo ""
echo "============================================================"
echo -e "  ${GREEN}✅ ENVIRONMENT CHECK COMPLETE${NC}"
echo "============================================================"
echo ""
echo "📋 Summary:"
echo "   • macOS: $MACOS_VERSION"
echo "   • Xcode: $XCODE_VERSION"
echo "   • Swift: $SWIFT_VERSION"
echo "   • SDK: $SDK_VERSION"
echo "   • CPU: $CPU_BRAND"
echo "   • RAM: ${RAM_GB}GB"
echo ""
echo "🚀 Ready to create Magellano Tahoe pipeline!"
echo ""
