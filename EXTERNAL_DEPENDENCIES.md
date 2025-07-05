# External Dependencies Management

This project includes code from two external GitHub repositories:

1. **AI_Diplomacy** - https://github.com/aiwaves-cn/AI_Diplomacy
2. **verifiers** - https://github.com/aiwaves-cn/verifiers

## Current Setup

The external repositories are currently included as **Git subtrees**, which allows easy updates while maintaining your own repository structure.

## Getting Updates from External Repositories

Since we're using Git subtrees, you can easily pull updates with simple commands:

### Simple Update Commands

```bash
# Update AI_Diplomacy
git subtree pull --prefix=AI_Diplomacy --squash https://github.com/EveryInc/AI_Diplomacy.git main

# Update verifiers  
git subtree pull --prefix=verifiers --squash https://github.com/willccbb/verifiers.git main

# Or use the convenience script
./update_external_deps.sh
```

### Alternative: Manual Update Process (Legacy)

If you need to update manually for any reason:

1. **For AI_Diplomacy:**
   ```bash
   git subtree pull --prefix=AI_Diplomacy --squash https://github.com/EveryInc/AI_Diplomacy.git main
   ```

2. **For verifiers:**
   ```bash
   git subtree pull --prefix=verifiers --squash https://github.com/willccbb/verifiers.git main
   ```

### Convert Back to Regular Directories (If Needed)

If you ever need to convert back to regular directories:

1. **Remove subtree tracking:**
   ```bash
   # Remove current directories
   git rm -r AI_Diplomacy verifiers
   git commit -m "Remove subtrees"
   
   # Clone as regular directories
   git clone https://github.com/EveryInc/AI_Diplomacy.git
   git clone https://github.com/willccbb/verifiers.git
   rm -rf AI_Diplomacy/.git verifiers/.git
   
   git add AI_Diplomacy verifiers
   git commit -m "Add external dependencies as regular directories"
   ```

## Repository Information

- **AI_Diplomacy Repository:** https://github.com/EveryInc/AI_Diplomacy
- **verifiers Repository:** https://github.com/willccbb/verifiers  
- **Your Repository:** https://github.com/OzDuys/diplomacy_self_play
- **Last Updated:** July 5, 2025 (converted to Git subtrees)

## Local Modifications

Document any local modifications you make to the external code here:

- [ ] AI_Diplomacy modifications: None yet
- [ ] verifiers modifications: None yet

## Notes

- Always backup your local changes before updating external dependencies
- Review the changes carefully to ensure they don't break your integration
- Test your project after updating external dependencies
- Update this document when you make local modifications to the external code
