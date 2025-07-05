#!/bin/bash

# Script to update external dependencies using Git subtrees
# Usage: ./update_external_deps.sh [ai_diplomacy|verifiers|all]

set -e

update_ai_diplomacy() {
    echo "Updating AI_Diplomacy using Git subtree..."
    
    git subtree pull --prefix=AI_Diplomacy --squash https://github.com/EveryInc/AI_Diplomacy.git main
    
    echo "AI_Diplomacy updated successfully!"
}

update_verifiers() {
    echo "Updating verifiers using Git subtree..."
    
    git subtree pull --prefix=verifiers --squash https://github.com/willccbb/verifiers.git main
    
    echo "verifiers updated successfully!"
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
echo "Update complete! The changes have been automatically committed."
echo "Don't forget to:"
echo "1. Review the changes with: git log --oneline -3"
echo "2. Test your integration"  
echo "3. Push to your repository: git push"
echo "4. Update EXTERNAL_DEPENDENCIES.md with the update date"
