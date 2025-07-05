"""
Module for constructing prompts for LLM interactions in the Diplomacy game.
"""

import logging
from typing import Dict, List, Optional, Any  # Added Any for game type placeholder

from config import config
from .utils import load_prompt, get_prompt_path
from .possible_order_context import (
    generate_rich_order_context,
    generate_rich_order_context_xml,
)
from .game_history import GameHistory  # Assuming GameHistory is correctly importable

# placeholder for diplomacy.Game to avoid circular or direct dependency if not needed for typehinting only
# from diplomacy import Game # Uncomment if 'Game' type hint is crucial and available

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Or inherit from parent logger

# --- Home-center lookup -------------------------------------------
HOME_CENTERS: dict[str, list[str]] = {
    "AUSTRIA": ["Budapest", "Trieste", "Vienna"],
    "ENGLAND": ["Edinburgh", "Liverpool", "London"],
    "FRANCE": ["Brest", "Marseilles", "Paris"],
    "GERMANY": ["Berlin", "Kiel", "Munich"],
    "ITALY": ["Naples", "Rome", "Venice"],
    "RUSSIA": ["Moscow", "Saint Petersburg", "Sevastopol", "Warsaw"],
    "TURKEY": ["Ankara", "Constantinople", "Smyrna"],
}


def build_context_prompt(
    game: Any,  # diplomacy.Game object
    board_state: dict,
    power_name: str,
    possible_orders: Dict[str, List[str]],
    game_history: GameHistory,
    agent_goals: Optional[List[str]] = None,
    agent_relationships: Optional[Dict[str, str]] = None,
    agent_private_diary: Optional[str] = None,
    prompts_dir: Optional[str] = None,
    include_messages: Optional[bool] = True,
) -> str:
    """Builds the detailed context part of the prompt.

    Args:
        game: The game object.
        board_state: Current state of the board.
        power_name: The name of the power for whom the context is being built.
        possible_orders: Dictionary of possible orders.
        game_history: History of the game (messages, etc.).
        agent_goals: Optional list of agent's goals.
        agent_relationships: Optional dictionary of agent's relationships with other powers.
        agent_private_diary: Optional string of agent's private diary.
        prompts_dir: Optional path to the prompts directory.

    Returns:
        A string containing the formatted context.
    """
    context_template = load_prompt("context_prompt.txt", prompts_dir=prompts_dir)

    # === Agent State Debug Logging ===
    if agent_goals:
        logger.debug(f"Using goals for {power_name}: {agent_goals}")
    if agent_relationships:
        logger.debug(f"Using relationships for {power_name}: {agent_relationships}")
    if agent_private_diary:
        logger.debug(f"Using private diary for {power_name}: {agent_private_diary[:200]}...")
    # ================================

    # Get our units and centers (not directly used in template, but good for context understanding)
    # units_info = board_state["units"].get(power_name, [])
    # centers_info = board_state["centers"].get(power_name, [])

    # Get the current phase
    year_phase = board_state["phase"]  # e.g. 'S1901M'

    # Decide which context builder to use.
    _use_simple = config.SIMPLE_PROMPTS
    if _use_simple:
        possible_orders_context_str = generate_rich_order_context(game, power_name, possible_orders)
    else:
        possible_orders_context_str = generate_rich_order_context_xml(game, power_name, possible_orders)

    if include_messages:
        messages_this_round_text = game_history.get_messages_this_round(power_name=power_name, current_phase_name=year_phase)
        if not messages_this_round_text.strip():
            messages_this_round_text = "\n(No messages this round)\n"
    else:
        messages_this_round_text = "\n"

    # Separate active and eliminated powers for clarity
    active_powers = [p for p in game.powers.keys() if not game.powers[p].is_eliminated()]
    eliminated_powers = [p for p in game.powers.keys() if game.powers[p].is_eliminated()]

    # Build units representation with power status
    units_lines = []
    for p, u in board_state["units"].items():
        u_str = ", ".join(u)
        if game.powers[p].is_eliminated():
            units_lines.append(f"  {p}: {u_str} [ELIMINATED]")
        else:
            units_lines.append(f"  {p}: {u_str}")
    units_repr = "\n".join(units_lines)

    # Build centers representation with power status
    centers_lines = []
    for p, c in board_state["centers"].items():
        c_str = ", ".join(c)
        if game.powers[p].is_eliminated():
            centers_lines.append(f"  {p}: {c_str} [ELIMINATED]")
        else:
            centers_lines.append(f"  {p}: {c_str}")
    centers_repr = "\n".join(centers_lines)

    # Build {home_centers}
    home_centers_str = ", ".join(HOME_CENTERS.get(power_name.upper(), []))

    order_history_str = game_history.get_order_history_for_prompt(
        game=game,  # Pass the game object for normalization
        power_name=power_name,
        current_phase_name=year_phase,
        num_movement_phases_to_show=1,
    )

    # Replace token only if it exists (template may not include it)
    if "{home_centers}" in context_template:
        context_template = context_template.replace("{home_centers}", home_centers_str)

    # Following the pattern for home_centers, use replace for safety
    if "{order_history}" in context_template:
        context_template = context_template.replace("{order_history}", order_history_str)

    context = context_template.format(
        power_name=power_name,
        current_phase=year_phase,
        all_unit_locations=units_repr,
        all_supply_centers=centers_repr,
        messages_this_round=messages_this_round_text,
        possible_orders=possible_orders_context_str,
        agent_goals="\n".join(f"- {g}" for g in agent_goals) if agent_goals else "None specified",
        agent_relationships="\n".join(f"- {p}: {s}" for p, s in agent_relationships.items()) if agent_relationships else "None specified",
        agent_private_diary=agent_private_diary if agent_private_diary else "(No diary entries yet)",
    )

    return context


def construct_order_generation_prompt(
    system_prompt: str,
    game: Any,  # diplomacy.Game object
    board_state: dict,
    power_name: str,
    possible_orders: Dict[str, List[str]],
    game_history: GameHistory,
    agent_goals: Optional[List[str]] = None,
    agent_relationships: Optional[Dict[str, str]] = None,
    agent_private_diary_str: Optional[str] = None,
    prompts_dir: Optional[str] = None,
) -> str:
    """Constructs the final prompt for order generation.

    Args:
        system_prompt: The base system prompt for the LLM.
        game: The game object.
        board_state: Current state of the board.
        power_name: The name of the power for whom the prompt is being built.
        possible_orders: Dictionary of possible orders.
        game_history: History of the game (messages, etc.).
        agent_goals: Optional list of agent's goals.
        agent_relationships: Optional dictionary of agent's relationships with other powers.
        agent_private_diary_str: Optional string of agent's private diary.
        prompts_dir: Optional path to the prompts directory.

    Returns:
        A string containing the complete prompt for the LLM.
    """
    # Load prompts
    _ = load_prompt("few_shot_example.txt", prompts_dir=prompts_dir)  # Loaded but not used, as per original logic
    # Pick the phase-specific instruction file (using unformatted versions)
    phase_code = board_state["phase"][-1]  # 'M' (movement), 'R', or 'A' / 'B'
    if phase_code == "M":
        instructions_file = get_prompt_path("order_instructions_movement_phase.txt")
    elif phase_code in ("A", "B"):  # builds / adjustments
        instructions_file = get_prompt_path("order_instructions_adjustment_phase.txt")
    elif phase_code == "R":  # retreats
        instructions_file = get_prompt_path("order_instructions_retreat_phase.txt")
    else:  # unexpected – default to movement rules
        instructions_file = get_prompt_path("order_instructions_movement_phase.txt")

    instructions = load_prompt(instructions_file, prompts_dir=prompts_dir)
    _use_simple = config.SIMPLE_PROMPTS

    # Build the context prompt
    context = build_context_prompt(
        game,
        board_state,
        power_name,
        possible_orders,
        game_history,
        agent_goals=agent_goals,
        agent_relationships=agent_relationships,
        agent_private_diary=agent_private_diary_str,
        prompts_dir=prompts_dir,
        include_messages=not _use_simple,  # include only when *not* simple
    )

    # Append goals at the end for focus
    goals_section = ""
    if agent_goals:
        goals_section = (
            "\n\nYOUR STRATEGIC GOALS:\n" + "\n".join(f"- {g}" for g in agent_goals) + "\n\nKeep these goals in mind when choosing your orders."
        )

    final_prompt = system_prompt + "\n\n" + context + "\n\n" + instructions + goals_section

    # Make the power names more LLM friendly
    final_prompt = (
        final_prompt.replace("AUSTRIA", "Austria")
        .replace("ENGLAND", "England")
        .replace("FRANCE", "France")
        .replace("GERMANY", "Germany")
        .replace("ITALY", "Italy")
        .replace("RUSSIA", "Russia")
        .replace("TURKEY", "Turkey")
    )
    logger.debug(f"Final order generation prompt preview for {power_name}: {final_prompt[:500]}...")

    return final_prompt
