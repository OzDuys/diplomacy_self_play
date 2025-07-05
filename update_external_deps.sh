#!/bin/bash

# Script to update external dependencies
# Usage: ./update_external_deps.sh [ai_diplomacy|verifiers|all]

set -e

update_ai_diplomacy() {
    echo "Updating AI_Diplomacy..."
    
    # Check if there are any local modifications
    if [ -d "AI_Diplomacy" ]; then
        echo "Backing up current AI_Diplomacy directory..."
        cp -r AI_Diplomacy AI_Diplomacy_backup_$(date +%Y%m%d_%H%M%S)
        
        # Remove current directory
        echo "Removing current AI_Diplomacy directory..."
        rm -rf AI_Diplomacy
    fi
    
    # Clone the latest version
    echo "Cloning latest AI_Diplomacy..."
    git clone https://github.com/aiwaves-cn/AI_Diplomacy.git
    
    # Remove .git directory
    echo "Removing .git directory..."
    rm -rf AI_Diplomacy/.git
    
    echo "AI_Diplomacy updated successfully!"
    echo "Please review changes and commit them with: git add AI_Diplomacy && git commit -m 'Update AI_Diplomacy to latest version'"
}

update_verifiers() {
    echo "Updating verifiers..."
    
    # Check if there are any local modifications
    if [ -d "verifiers" ]; then
        echo "Backing up current verifiers directory..."
        cp -r verifiers verifiers_backup_$(date +%Y%m%d_%H%M%S)
        
        # Remove current directory
        echo "Removing current verifiers directory..."
        rm -rf verifiers
    fi
    
    # Clone the latest version
    echo "Cloning latest verifiers..."
    git clone https://github.com/aiwaves-cn/verifiers.git
    
    # Remove .git directory
    echo "Removing .git directory..."
    rm -rf verifiers/.git
    
    echo "verifiers updated successfully!"
    echo "Please review changes and commit them with: git add verifiers && git commit -m 'Update verifiers to latest version'"
}

# Main script logic
case "$1" in
    "ai_diplomacy")
        update_ai_diplomacy
        ;;
    "verifiers")
        update_verifiers
        ;;
    "all"|"")
        update_ai_diplomacy
        update_verifiers
        ;;
    *)
        echo "Usage: $0 [ai_diplomacy|verifiers|all]"
        echo "  ai_diplomacy  - Update only AI_Diplomacy"
        echo "  verifiers     - Update only verifiers"
        echo "  all           - Update both (default)"
        exit 1
        ;;
esac

echo ""
echo "Update complete! Don't forget to:"
echo "1. Review the changes"
echo "2. Test your integration"
echo "3. Commit the updates to your repository"
echo "4. Update EXTERNAL_DEPENDENCIES.md with the update date"
