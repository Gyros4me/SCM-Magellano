#!/bin/bash
# SCMMagellano Project Tree Generator
# Genera una struttura ad albero del progetto Swift

set -e

PROJECT_ROOT="/Volumes/Audio/masteraudio/Developer/SCMMagellano"
OUTPUT_FILE="project_structure.txt"

echo "🌳 Generating SCMMagellano Project Tree..."
echo "Project: $PROJECT_ROOT"
echo "Date: $(date)"
echo ""

# Function to generate tree with proper formatting
generate_tree() {
    local dir="$1"
    local prefix="$2"
    
    # Get all items in directory, excluding hidden files
    local items=()
    while IFS= read -r -d $'\0' item; do
        items+=("$item")
    done < <(find "$dir" -maxdepth 1 -mindepth 1 -not -name ".*" -not -name "*.xcodeproj" -not -name "*.xcworkspace" -print0 | sort -z)
    
    local count=${#items[@]}
    local i=0
    
    for item in "${items[@]}"; do
        i=$((i + 1))
        local name=$(basename "$item")
        
        # Determine the prefix characters
        if [ $i -eq $count ]; then
            echo "${prefix}└── $name"
            local new_prefix="${prefix}    "
        else
            echo "${prefix}├── $name"
            local new_prefix="${prefix}│   "
        fi
        
        # If it's a directory, recurse
        if [ -d "$item" ]; then
            # Check if directory contains Swift files or other relevant content
            local has_content=$(find "$item" -name "*.swift" -o -name "*.metal" -o -name "*.json" -o -name "*.plist" | head -1)
            if [ -n "$has_content" ]; then
                generate_tree "$item" "$new_prefix"
            else
                echo "${new_prefix}    (empty)"
            fi
        fi
    done
}

# Function to count Swift files and lines
count_swift_metrics() {
    echo ""
    echo "📊 Swift Files Metrics:"
    echo "======================"
    
    local total_files=$(find "$PROJECT_ROOT" -name "*.swift" -type f | wc -l | tr -d ' ')
    local total_lines=$(find "$PROJECT_ROOT" -name "*.swift" -type f -exec wc -l {} + | tail -1 | awk '{print $1}')
    
    echo "Total Swift Files: $total_files"
    echo "Total Lines of Code: $total_lines"
    echo ""
    
    # Breakdown by directory
    echo "Breakdown by Directory:"
    find "$PROJECT_ROOT" -name "*.swift" -type f | sed 's|/[^/]*$||' | sort | uniq -c | sort -nr | while read count dir; do
        echo "  $count files: $dir"
    done
}

# Function to list all Swift files with paths
list_swift_files() {
    echo ""
    echo "📝 All Swift Files:"
    echo "==================="
    
    find "$PROJECT_ROOT" -name "*.swift" -type f | sort | while read file; do
        local relative_path="${file#$PROJECT_ROOT/}"
        local lines=$(wc -l < "$file" | tr -d ' ')
        echo "  📄 $relative_path ($lines lines)"
    done
}

# Function to show file sizes
show_file_sizes() {
    echo ""
    echo "💾 File Sizes:"
    echo "=============="
    
    find "$PROJECT_ROOT" -type f \( -name "*.swift" -o -name "*.metal" \) -exec du -h {} + | sort -hr | head -10 | while read size file; do
        local relative_path="${file#$PROJECT_ROOT/}"
        echo "  $size: $relative_path"
    done
}

# Main execution
cd "$PROJECT_ROOT"

# Generate the tree structure
echo "🏗️  Project Structure:"
echo "====================="
generate_tree "$PROJECT_ROOT" ""

# Additional metrics
count_swift_metrics
list_swift_files
show_file_sizes

# Generate summary
echo ""
echo "🎯 Project Summary:"
echo "==================="
echo "Project Name: SCMMagellano"
echo "Root: $PROJECT_ROOT"
echo "Type: Swift Package Manager Project"
echo "Architecture: Mamba-MoE Hybrid with QLoRA"
echo "Target: 3.3B Parameter Model on Apple Silicon"
echo ""
echo "📁 Main Components:"
echo "  • MagellanoCLI - Command line interface"
echo "  • MagellanoCore - Core model architecture"
echo "  • MagellanoMetal - GPU acceleration kernels"
echo ""
echo "🔧 Key Features:"
echo "  • NF4 Quantization (86.5% memory reduction)"
echo "  • QLoRA Fine-tuning"
echo "  • Metal GPU Optimization"
echo "  • Mamba SSM + Mixture of Experts"
echo "  • Dynamic Memory Management"

# Save to file
{
    echo "SCMMagellano Project Structure"
    echo "Generated: $(date)"
    echo "=========================================="
    echo ""
    generate_tree "$PROJECT_ROOT" ""
    count_swift_metrics
    list_swift_files
} > "$OUTPUT_FILE"

echo ""
echo "✅ Project structure saved to: $OUTPUT_FILE"