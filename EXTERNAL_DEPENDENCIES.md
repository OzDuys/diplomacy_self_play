# External Dependencies Management

This project includes code from two external GitHub repositories:

1. **AI_Diplomacy** - https://github.com/aiwaves-cn/AI_Diplomacy
2. **verifiers** - https://github.com/aiwaves-cn/verifiers

## Current Setup

The external repositories are currently included as direct copies in the `AI_Diplomacy/` and `verifiers/` directories.

## Getting Updates from External Repositories

Since we've included these as direct copies rather than git subtrees, here's how to manually update them:

### Option 1: Manual Update Process

1. **For AI_Diplomacy:**
   ```bash
   # Backup any local changes you made
   cp -r AI_Diplomacy AI_Diplomacy_backup
   
   # Remove the current directory
   rm -rf AI_Diplomacy
   
   # Clone the latest version
   git clone https://github.com/aiwaves-cn/AI_Diplomacy.git
   
   # Remove the .git directory to prevent conflicts
   rm -rf AI_Diplomacy/.git
   
   # Review changes and restore any local modifications
   # Then commit the updates
   git add AI_Diplomacy
   git commit -m "Update AI_Diplomacy to latest version"
   ```

2. **For verifiers:**
   ```bash
   # Backup any local changes you made
   cp -r verifiers verifiers_backup
   
   # Remove the current directory
   rm -rf verifiers
   
   # Clone the latest version
   git clone https://github.com/aiwaves-cn/verifiers.git
   
   # Remove the .git directory to prevent conflicts
   rm -rf verifiers/.git
   
   # Review changes and restore any local modifications
   # Then commit the updates
   git add verifiers
   git commit -m "Update verifiers to latest version"
   ```

### Option 2: Convert to Git Subtrees (Recommended for Future)

If you want to use Git subtrees for easier management, you can convert the current setup:

1. **First, remove the current directories and set up subtrees:**
   ```bash
   # Remove current directories
   git rm -r AI_Diplomacy verifiers
   git commit -m "Remove directories before converting to subtrees"
   
   # Add as subtrees
   git subtree add --prefix=AI_Diplomacy --squash https://github.com/aiwaves-cn/AI_Diplomacy.git main
   git subtree add --prefix=verifiers --squash https://github.com/aiwaves-cn/verifiers.git main
   ```

2. **Then to update in the future:**
   ```bash
   # Update AI_Diplomacy
   git subtree pull --prefix=AI_Diplomacy --squash https://github.com/aiwaves-cn/AI_Diplomacy.git main
   
   # Update verifiers
   git subtree pull --prefix=verifiers --squash https://github.com/aiwaves-cn/verifiers.git main
   ```

## Repository Information

- **AI_Diplomacy Repository:** https://github.com/aiwaves-cn/AI_Diplomacy
- **verifiers Repository:** https://github.com/aiwaves-cn/verifiers
- **Your Repository:** https://github.com/OzDuys/diplomacy_self_play
- **Last Updated:** July 5, 2025 (initial inclusion)

## Local Modifications

Document any local modifications you make to the external code here:

- [ ] AI_Diplomacy modifications: None yet
- [ ] verifiers modifications: None yet

## Notes

- Always backup your local changes before updating external dependencies
- Review the changes carefully to ensure they don't break your integration
- Test your project after updating external dependencies
- Update this document when you make local modifications to the external code
