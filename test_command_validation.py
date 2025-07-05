#!/usr/bin/env python3
"""
Quick command validation test for AI_Diplomacy integration.
"""

import sys
import tempfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_command_generation():
    """Test that we generate valid AI_Diplomacy commands."""
    from diplomacy_grpo.integration.ai_diplomacy_env import DiplomacyGRPOEnvironment
    
    with tempfile.TemporaryDirectory() as temp_dir:
        env = DiplomacyGRPOEnvironment(
            max_year=1902,
            num_negotiation_rounds=0,
            working_dir=temp_dir
        )
        
        # Test model assignment
        assignments = env._create_model_assignment('Austria', 'gpt-4o-mini')
        print(f"✓ Model assignments: {assignments}")
        
        # Test command building (without execution)
        import subprocess
        import sys
        
        game_id = "test_game"
        game_dir = env.working_dir / game_id
        game_dir.mkdir(exist_ok=True)
        
        ai_diplomacy_path = Path(__file__).parent / "AI_Diplomacy"
        lm_game_path = ai_diplomacy_path / "lm_game.py"
        
        country_order = ['AUSTRIA', 'ENGLAND', 'FRANCE', 'GERMANY', 'ITALY', 'RUSSIA', 'TURKEY']
        model_list = [assignments[country.title()] for country in country_order]
        
        cmd = [
            sys.executable, 
            str(lm_game_path),
            f"--run_dir={game_dir}",
            f"--max_year={env.max_year}", 
            f"--num_negotiation_rounds={env.num_negotiation_rounds}",
            f"--models={','.join(model_list)}",
            "--generate_phase_summaries=false",
            "--help"  # Just show help to validate args
        ]
        
        print(f"✓ Command: {' '.join(cmd)}")
        
        # Test command validation (just check help)
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("✓ Command arguments are valid")
            return True
        else:
            print(f"❌ Command failed: {result.stderr}")
            return False

if __name__ == "__main__":
    print("🔧 Testing AI_Diplomacy Command Generation")
    
    try:
        success = test_command_generation()
        if success:
            print("✅ Command validation passed!")
        else:
            print("❌ Command validation failed!")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)