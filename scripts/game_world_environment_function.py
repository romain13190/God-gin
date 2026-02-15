import os
import re
import random
import requests
from itertools import combinations
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Semaphore
from trl.experimental.openenv import generate_rollout_completions


# ============================================================
# Card Utilities
# ============================================================
RANK_CHARS = "A23456789TJQK"
SUIT_CHARS = "scdh"
RANK_TO_IDX = {c: i for i, c in enumerate(RANK_CHARS)}
SUIT_TO_IDX = {c: i for i, c in enumerate(SUIT_CHARS)}


def card_name_to_id(name):
    """Convert card name like '3s' to card ID (0-51). suit*13 + rank."""
    return SUIT_TO_IDX[name[1]] * 13 + RANK_TO_IDX[name[0]]


def card_id_to_name(cid):
    """Convert card ID to name like '3s'."""
    return RANK_CHARS[cid % 13] + SUIT_CHARS[cid // 13]


def card_rank(cid):
    return cid % 13


def card_suit(cid):
    return cid // 13


def card_dw_value(cid):
    """Deadwood value: A=1, 2-10=face value, J=Q=K=10 (standard gin rummy)."""
    return min(card_rank(cid) + 1, 10)


# ============================================================
# Observation Parsing
# ============================================================
def format_observation(obs_text):
    """Format observation for Gin Rummy. Passthrough - env already formats well."""
    return obs_text


def parse_player_id(obs_text):
    m = re.search(r'You are Player (\d+)', obs_text)
    return int(m.group(1)) if m else 0


def parse_hand_from_obs(obs_text):
    """Parse hand cards from observation grid. Returns list of card IDs."""
    pid = parse_player_id(obs_text)
    pattern = rf'Player{pid}: Deadwood=\d+\s*\n\+-+\+\n((?:\|.*\n)+)\+-+\+'
    grid_match = re.search(pattern, obs_text)
    if not grid_match:
        return []
    grid_text = grid_match.group(1)
    card_names = re.findall(r'[A23456789TJQK][shdc]', grid_text)
    return [card_name_to_id(n) for n in card_names]


def parse_legal_actions(obs_text):
    """Parse legal action IDs from observation text."""
    actions = re.findall(r'^\s*(\d+)\s*->', obs_text, re.MULTILINE)
    return [int(a) for a in actions]


def parse_upcard_id(obs_text):
    """Parse upcard card ID from observation."""
    m = re.search(r'Upcard:\s*([A23456789TJQK][shdc])', obs_text)
    return card_name_to_id(m.group(1)) if m else None


def parse_deadwood_from_obs(obs_text):
    """Parse current deadwood value from observation."""
    pid = parse_player_id(obs_text)
    m = re.search(rf'Player{pid}: Deadwood=(\d+)', obs_text)
    return int(m.group(1)) if m else None


# ============================================================
# Meld Finding & Deadwood Computation
# ============================================================
def find_all_melds(hand):
    """Find all possible melds (sets and runs) in a hand."""
    melds = []

    # Sets: 3 or 4 cards of same rank
    by_rank = {}
    for c in hand:
        by_rank.setdefault(card_rank(c), []).append(c)
    for r, cards in by_rank.items():
        if len(cards) >= 3:
            for combo in combinations(cards, 3):
                melds.append(frozenset(combo))
            if len(cards) >= 4:
                melds.append(frozenset(cards))

    # Runs: 3+ consecutive cards of same suit
    by_suit = {}
    for c in hand:
        by_suit.setdefault(card_suit(c), []).append(c)
    for s, cards in by_suit.items():
        ranks = sorted(set(card_rank(c) for c in cards))
        i = 0
        while i < len(ranks):
            run = [ranks[i]]
            j = i + 1
            while j < len(ranks) and ranks[j] == run[-1] + 1:
                run.append(ranks[j])
                j += 1
            if len(run) >= 3:
                for start in range(len(run)):
                    for end in range(start + 3, len(run) + 1):
                        meld = frozenset(s * 13 + r for r in run[start:end])
                        melds.append(meld)
            i = j if j > i + 1 else i + 1

    return melds


def compute_min_deadwood(hand):
    """Compute minimum deadwood for a hand by finding optimal non-overlapping melds."""
    if not hand:
        return 0

    total_dw = sum(card_dw_value(c) for c in hand)
    melds = find_all_melds(hand)

    if not melds:
        return total_dw

    # Sort melds by value descending for better pruning
    meld_values = [sum(card_dw_value(c) for c in m) for m in melds]
    order = sorted(range(len(melds)), key=lambda i: meld_values[i], reverse=True)
    melds = [melds[i] for i in order]
    meld_values = [meld_values[i] for i in order]

    best = [total_dw]

    def search(idx, used, melded_dw):
        remaining = total_dw - melded_dw
        if remaining < best[0]:
            best[0] = remaining
        if best[0] == 0 or idx >= len(melds):
            return
        # Pruning: max possible remaining meld value
        max_possible = sum(meld_values[i] for i in range(idx, len(melds)))
        if remaining - max_possible >= best[0]:
            return
        for i in range(idx, len(melds)):
            m = melds[i]
            if not (m & used):
                search(i + 1, used | m, melded_dw + meld_values[i])

    search(0, frozenset(), 0)
    return best[0]


# ============================================================
# Optimal Action Computation
# ============================================================
def compute_optimal_action(obs_text):
    """
    Determine optimal action from observation text.
    Strategy: knock > meld declare > layoff > smart draw > smart discard > pass
    Returns action ID or None.
    """
    legal_actions = parse_legal_actions(obs_text)
    if not legal_actions:
        return None
    if len(legal_actions) == 1:
        return legal_actions[0]

    # KNOCK always (highest priority)
    if 55 in legal_actions:
        return 55

    # Meld declarations (after knock): declare all available
    meld_actions = [a for a in legal_actions if a >= 56]
    if meld_actions:
        return meld_actions[0]

    # Check phase for layoff handling
    phase_match = re.search(r'Phase:\s*(\w+)', obs_text)
    phase = phase_match.group(1) if phase_match else ""

    if phase == "Layoff":
        layoff = [a for a in legal_actions if a < 52]
        if layoff:
            return layoff[0]
        if 54 in legal_actions:
            return 54
        return legal_actions[0]

    hand = parse_hand_from_obs(obs_text)

    # DRAW phase (52=upcard + 53=stock both available)
    if 52 in legal_actions and 53 in legal_actions:
        upcard = parse_upcard_id(obs_text)
        if upcard is not None and hand:
            # Simulate taking upcard: add to hand, try each discard
            hand_plus = hand + [upcard]
            best_dw_upcard = 999
            for c in hand_plus:
                trial = list(hand_plus)
                trial.remove(c)
                dw = compute_min_deadwood(trial)
                best_dw_upcard = min(best_dw_upcard, dw)

            current_dw = compute_min_deadwood(hand)
            if best_dw_upcard < current_dw:
                return 52  # Take upcard - it helps
        return 53  # Draw stock

    # FIRST UPCARD phase (52=take + 54=pass, no stock option)
    if 54 in legal_actions and 52 in legal_actions and 53 not in legal_actions:
        upcard = parse_upcard_id(obs_text)
        if upcard is not None and hand:
            hand_plus = hand + [upcard]
            best_dw_take = 999
            for c in hand_plus:
                trial = list(hand_plus)
                trial.remove(c)
                dw = compute_min_deadwood(trial)
                best_dw_take = min(best_dw_take, dw)

            current_dw = compute_min_deadwood(hand)
            if best_dw_take < current_dw:
                return 52  # Take upcard
        return 54  # Pass

    # DISCARD phase (actions 0-51 = card IDs)
    discard_actions = [a for a in legal_actions if a < 52]
    if discard_actions and hand:
        best_action = discard_actions[0]
        best_dw = 999
        for a in discard_actions:
            trial = list(hand)
            if a in trial:
                trial.remove(a)
            else:
                continue
            dw = compute_min_deadwood(trial)
            if dw < best_dw or (dw == best_dw and card_dw_value(a) > card_dw_value(best_action)):
                best_dw = dw
                best_action = a
        return best_action

    # Fallback: pass or first legal action
    if 54 in legal_actions:
        return 54
    return legal_actions[0]


# ============================================================
# System Prompts
# ============================================================
GIN_RUMMY_SYSTEM_PROMPT = (
    "You are playing Gin Rummy.\n\n"
    "# Game Rules\n"
    "GIN RUMMY:\n"
    "- 52-card deck, each player receives 10 cards\n"
    "- Form MELDS (sets of 3+ same rank, or runs of 3+ consecutive same suit) to minimize DEADWOOD\n"
    "- Card values: A=1, 2-10=face value, J=Q=K=10\n"
    "- Card notation: A,2-9,T,J,Q,K + s(spades),h(hearts),d(diamonds),c(clubs). Example: Th=10 of hearts\n\n"
    "EACH TURN:\n"
    "1. DRAW: Pick from stock pile (action 53) or upcard/discard pile (action 52)\n"
    "2. If deadwood <= knock limit: you may KNOCK (action 55) to end the hand\n"
    "3. If not knocking: DISCARD one card (action = card's ID number)\n\n"
    "FIRST UPCARD: At game start, take first upcard (52) or pass (54)\n"
    "LAYOFF: After opponent knocks, lay off cards onto their melds or pass (54)\n\n"
    "# Output Format\n"
    "You must respond with ONLY the action ID (a single number).\n"
    "Do NOT include descriptions or explanations.\n\n"
    "Examples:\n"
    "- To draw from stock: respond \"53\"\n"
    "- To knock: respond \"55\"\n"
    "- To discard a card: respond with its ID number"
)

GIN_RUMMY_HINT = (
    "\n\nSTRATEGY HINTS:\n"
    "1. ALWAYS KNOCK when you can (action 55) - this is the highest priority move\n"
    "2. For DRAW: Take the upcard (52) ONLY if adding it to your hand and discarding optimally "
    "gives lower deadwood than your current hand. Otherwise draw from stock (53)\n"
    "3. For DISCARD: Discard the card that leaves your hand with the lowest possible deadwood. "
    "Prefer discarding high-value cards that don't contribute to melds\n"
    "4. For FIRST UPCARD: Take it (52) only if it clearly helps form a meld. Otherwise pass (54)\n"
    "5. Focus on forming melds (sets of same rank, runs of consecutive same suit)"
)


# ============================================================
# Reasoning Tag Removal
# ============================================================
REASONING_TAG_PAIRS = [
    ("think", "think"),
    ("thinking", "thinking"),
    ("reasoning", "reasoning"),
    ("thought", "thought"),
    ("reflection", "reflection"),
]

def remove_reasoning_tags(text: str) -> str:

    cleaned = text

    for tag_name, close_name in REASONING_TAG_PAIRS:
        cleaned = re.sub(
            rf"<{tag_name}>.*?</{close_name}>",
            "",
            cleaned,
            flags=re.DOTALL | re.IGNORECASE,
        )

        close_tag = f"</{close_name}>"
        if close_tag in cleaned:
            cleaned = cleaned.split(close_tag)[-1]

        open_match = re.search(rf"<{tag_name}>", cleaned, flags=re.IGNORECASE)
        if open_match:
            cleaned = cleaned[: open_match.start()]

    cleaned = re.sub(r"\n\s*\n\s*\n", "\n\n", cleaned)
    return cleaned.strip()


# ============================================================
# Curriculum Scheduler
# ============================================================
class CurriculumScheduler:
    """
    Manages curriculum learning parameters throughout training.
    """
    def __init__(
        self,
        initial_max_turn=1,
        final_max_turn=50,
        rollouts_per_stage=1280,
        initial_hint_prob=0.75,
        final_hint_prob=0.0,
        warmup_rollouts=128,
    ):
        self.initial_max_turn = initial_max_turn
        self.final_max_turn = final_max_turn
        self.rollouts_per_stage = rollouts_per_stage
        self.initial_hint_prob = initial_hint_prob
        self.final_hint_prob = final_hint_prob
        self.warmup_rollouts = warmup_rollouts

        self.total_rollouts = 0

    def get_max_turn(self):
        """Calculate current max_turn based on curriculum."""
        if self.total_rollouts < self.warmup_rollouts:
            return self.initial_max_turn

        adjusted_rollouts = self.total_rollouts - self.warmup_rollouts
        stage = adjusted_rollouts // self.rollouts_per_stage

        current_max_turn = min(
            self.initial_max_turn + stage,
            self.final_max_turn
        )
        return current_max_turn

    def get_hint_prob(self):
        """Calculate current hint probability based on curriculum."""
        if self.total_rollouts < self.warmup_rollouts:
            return self.initial_hint_prob

        total_stages = self.final_max_turn - self.initial_max_turn
        total_decay_rollouts = total_stages * self.rollouts_per_stage

        adjusted_rollouts = self.total_rollouts - self.warmup_rollouts
        progress = min(adjusted_rollouts / total_decay_rollouts, 1.0)

        current_prob = self.initial_hint_prob - progress * (self.initial_hint_prob - self.final_hint_prob)
        return max(current_prob, self.final_hint_prob)

    def step(self, num_rollouts=1):
        """Increment rollout counter."""
        self.total_rollouts += num_rollouts

    def get_status(self):
        """Get current curriculum status for logging."""
        return {
            "total_rollouts": self.total_rollouts,
            "max_turn": self.get_max_turn(),
            "hint_prob": self.get_hint_prob(),
        }


# ============================================================
# Rollout Function 1: Simple (no strategy forcing)
# ============================================================
def rollout_first_prompt_and_completion(prompts: list[str], trainer, max_turns: int = 30) -> dict[str, list]:
    from trl.experimental.openenv import generate_rollout_completions
    import os
    import random
    import requests
    import json

    games_to_task_id_range = {
        "goofspiel": (0, 99999999),
        "liars_dice": (100000000, 199999999),
        "leduc_poker": (200000000, 299999999),
        "gin_rummy": (300000000, 399999999),
        "othello": (400000000, 499999999),
        "backgammon": (500000000, 599999999),
        "hex": (600000000, 699999999),
        "clobber": (700000000, 799999999),
    }

    selected_game = "gin_rummy"

    # --- 1. Static Initialization (Once per Rank) ---
    if not getattr(rollout_first_prompt_and_completion, "initialized", False):
        rank = int(os.environ.get("LOCAL_RANK", "0"))

        raw_urls = os.environ.get("ENVIRONMENT_SERVER_URLS", "")
        server_list = [url.strip() for url in raw_urls.split(",") if url.strip()]

        if not server_list:
            base_url = ""
            print("Warning: No ENVIRONMENT_SERVER_URLS found.")
        else:
            base_url = server_list[rank % len(server_list)]

        rollout_first_prompt_and_completion.base_url = base_url

        try:
            print(f"Initializing environment on rank {rank} at {base_url}...")
            payload = {"task_id": games_to_task_id_range[selected_game][0], "seed": 42, "opponent": "mcts"}
            create_res = requests.post(f"{base_url}/reset", json=payload, timeout=300)
            create_res.raise_for_status()
            rollout_first_prompt_and_completion.initialized = True
            print(f"Environment initialized. Rank: {rank}.")
        except Exception as e:
            print(f"CRITICAL: Failed to create environment on rank {rank}: {e}")
            raise e

    env_endpoint = rollout_first_prompt_and_completion.base_url

    # --- 2. Rollout Setup ---
    all_episode_prompt_ids: list[list[int]] = []
    all_episode_completion_ids: list[list[int]] = []
    all_episode_logprobs: list[list[float]] = []
    all_episode_rewards: list[float] = []

    tokenizer = trainer.processing_class
    TIMEOUT = 2400

    # --- 3. Batch Loop ---
    game_id = random.randint(games_to_task_id_range[selected_game][0], games_to_task_id_range[selected_game][1])

    for i, prompt in enumerate(prompts):
        episode_prompt_ids: list[int] = []
        episode_completion_ids: list[int] = []
        episode_logprobs: list[float] = []
        done = False
        solved = False
        train_reward = 0
        turn_number = 0

        # --- Reset Environment (POST /reset) ---
        payload = {"task_id": game_id, "seed": 42, "opponent": "mcts"}

        try:
            reset_res = requests.post(f"{env_endpoint}/reset", json=payload, timeout=TIMEOUT)
            reset_res.raise_for_status()
            reset_data = reset_res.json()
            result_block = reset_data["result"]

            episode_id = result_block.get("episode_id", "")

            current_observation = result_block.get("observation", "")
            format_instructions = 'Your output must strictly follow this format: "Thought:\nyour thoughts ONLY in text.\n\nAction:\nONLY your action ID (a single number)."'
            current_observation += format_instructions

        except Exception as e:
            print(f"Failed to reset environment (Game {game_id}): {e}")
            continue

        # --- Build Conversation History ---
        messages = []

        messages.append({"role": "user", "content": current_observation})

        # --- Interaction Loop ---
        while not done and (turn_number < max_turns):
            rollout_outputs = generate_rollout_completions(trainer, prompts=[messages], as_chat=True)[0]
            prompt_ids = rollout_outputs.get("prompt_ids", [])
            completion_ids = rollout_outputs.get("completion_ids", [])
            logprobs = rollout_outputs.get("logprobs", [])
            completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()

            if turn_number == 0:
                episode_prompt_ids = prompt_ids
                episode_completion_ids = completion_ids
                episode_logprobs = logprobs

            messages.append({"role": "assistant", "content": completion_text})

            # --- Parse Action ---
            action_to_send = completion_text
            if action_to_send.endswith("</s>"):
                action_to_send = action_to_send[:-5]

            if "Action:" in action_to_send:
                action_to_send = action_to_send.split("Action:")[-1].strip()

            # --- Step Environment (POST /step) ---
            try:
                formatted_observation = ""
                step_payload = {"action": action_to_send, "episode_id": episode_id}
                step_res = requests.post(f"{env_endpoint}/step", json=step_payload, timeout=TIMEOUT)
                step_res.raise_for_status()
                step_data = step_res.json()
                step_block = step_data["result"]

                step_state = step_block.get("observation", "")
                step_reward = step_block.get("reward", 0)
                done = step_block.get("done", False)

                formatted_observation = step_state

            except Exception as e:
                print(f"Step failed: {e}")
                formatted_observation = "Invalid Action.\n\n" + formatted_observation
                step_reward = -0.01
                done = False

            if done:
                train_reward = step_reward
            else:
                messages.append({"role": "user", "content": formatted_observation})

            turn_number += 1

        all_episode_prompt_ids.append(episode_prompt_ids)
        all_episode_completion_ids.append(episode_completion_ids)
        all_episode_logprobs.append(episode_logprobs)
        all_episode_rewards.append(train_reward)

    return {
        "prompt_ids": all_episode_prompt_ids,
        "completion_ids": all_episode_completion_ids,
        "logprobs": all_episode_logprobs,
        "env_rewards": all_episode_rewards
    }


# ============================================================
# Rollout Function 2: Last-turn training with strategy forcing
# ============================================================
def rollout_last_prompt_and_completion_parallelized_curriculum(
    prompts: list[str],
    trainer,
    max_turns: int = 30,
) -> dict[str, list]:
    """
    Parallelized rollout function for Gin Rummy.
    Uses strategy forcing for early turns, trains model on one target turn.
    """
    # --- Constants ---
    STRATEGY_REWARD = 1.0
    INVALID_PENALTY = -0.1

    games_to_task_id_range = {
        "goofspiel": (0, 99999999),
        "liars_dice": (100000000, 199999999),
        "leduc_poker": (200000000, 299999999),
        "gin_rummy": (300000000, 399999999),
        "othello": (400000000, 499999999),
        "backgammon": (500000000, 599999999),
        "hex": (600000000, 699999999),
        "clobber": (700000000, 799999999),
    }

    selected_game = "gin_rummy"

    # --- 1. Static Initialization (Once per Rank) ---
    if not getattr(rollout_last_prompt_and_completion_parallelized_curriculum, "initialized", False):
        rank = int(os.environ.get("LOCAL_RANK", "0"))
        raw_urls = os.environ.get("ENVIRONMENT_SERVER_URLS", "")
        server_urls = [u.strip() for u in raw_urls.split(",") if u.strip()]

        if not server_urls:
            raise RuntimeError("ENVIRONMENT_SERVER_URLS is empty")

        env_pool = []

        for idx, base_url in enumerate(server_urls):
            try:
                print(f"[INIT] Initializing env on server {idx}: {base_url}")
                payload = {"task_id": games_to_task_id_range[selected_game][0], "seed": 42, "opponent": "mcts"}
                res = requests.post(f"{base_url}/reset", json=payload, timeout=300)
                res.raise_for_status()
                env_pool.append({"base_url": base_url})
                print(f"[INIT] Server {idx} ready")
            except Exception as e:
                raise RuntimeError(f"Failed to init server {base_url}: {e}")

        rollout_last_prompt_and_completion_parallelized_curriculum.rank = rank
        rollout_last_prompt_and_completion_parallelized_curriculum.env_pool = env_pool
        rollout_last_prompt_and_completion_parallelized_curriculum.num_servers = len(env_pool)
        rollout_last_prompt_and_completion_parallelized_curriculum.initialized = True
        rollout_last_prompt_and_completion_parallelized_curriculum.thread_pool = ThreadPoolExecutor(max_workers=len(env_pool))
        rollout_last_prompt_and_completion_parallelized_curriculum.generation_semaphore = Semaphore(1)
        rollout_last_prompt_and_completion_parallelized_curriculum.games_to_task_id_range = games_to_task_id_range
        rollout_last_prompt_and_completion_parallelized_curriculum.selected_game = selected_game

        # Initialize curriculum scheduler
        rollout_last_prompt_and_completion_parallelized_curriculum.curriculum = CurriculumScheduler(
            initial_max_turn=trainer.args.initial_max_turn,
            final_max_turn=50,
            rollouts_per_stage=trainer.args.rollouts_per_stage,
            initial_hint_prob=0.75,
            final_hint_prob=0.0,
            warmup_rollouts=trainer.args.rollouts_per_stage,
        )
        print(f"[CURRICULUM] Initialized with initial_max_turn={trainer.args.initial_max_turn}, final_max_turn=50, rollouts_per_stage={trainer.args.rollouts_per_stage}, initial_hint_prob=0.75, final_hint_prob=0.0, warmup_rollouts={trainer.args.rollouts_per_stage}")

    # Retrieve static variables
    rank = rollout_last_prompt_and_completion_parallelized_curriculum.rank
    env_pool = rollout_last_prompt_and_completion_parallelized_curriculum.env_pool
    num_servers = rollout_last_prompt_and_completion_parallelized_curriculum.num_servers
    games_to_task_id_range = rollout_last_prompt_and_completion_parallelized_curriculum.games_to_task_id_range
    selected_game = rollout_last_prompt_and_completion_parallelized_curriculum.selected_game
    curriculum = rollout_last_prompt_and_completion_parallelized_curriculum.curriculum

    tokenizer = trainer.processing_class
    TIMEOUT = 2400

    # Get current curriculum parameters
    total_rollouts = curriculum.total_rollouts
    current_max_turn = curriculum.get_max_turn()
    current_hint_prob = curriculum.get_hint_prob()
    print(f"[CURRICULUM] Rollout {total_rollouts}: max_turn={current_max_turn}, hint_prob={current_hint_prob:.2f}")

    def run_single_prompt(index: int, prompt: str):
        game_id = int(prompt)

        server_idx = (index + rank) % num_servers
        server = env_pool[server_idx]
        env_endpoint = server["base_url"]
        done = False
        turn_number = 0
        target_training_turn = current_max_turn - 1

        # Determine if this episode gets hints
        use_hints = random.random() < current_hint_prob

        # --- Reset Environment (POST /reset) ---
        payload = {"task_id": game_id, "seed": 42, "opponent": "mcts"}

        try:
            reset_res = requests.post(f"{env_endpoint}/reset", json=payload, timeout=TIMEOUT)
            reset_res.raise_for_status()
            reset_data = reset_res.json()
            result_block = reset_data["result"]

            episode_id = result_block.get("episode_id", "")

            raw_observation = result_block.get("observation", "")
            formatted_observation = format_observation(raw_observation)

        except Exception as e:
            print(f"Failed to reset environment (Game {game_id}): {e}")
            return index, None

        # --- Build Conversation History ---
        system_prompt = GIN_RUMMY_SYSTEM_PROMPT

        if use_hints:
            system_prompt += GIN_RUMMY_HINT

        messages = [{"role": "system", "content": system_prompt}]

        # Strategy forcing for turns before target training turn
        while not done and (turn_number < target_training_turn):
            messages.append({"role": "user", "content": formatted_observation})

            optimal_action = compute_optimal_action(formatted_observation)
            if optimal_action is None:
                target_training_turn = turn_number
                break

            messages.append({"role": "assistant", "content": str(optimal_action)})

            # --- Step Environment (POST /step) ---
            try:
                formatted_observation = ""
                step_payload = {"action": str(optimal_action), "episode_id": episode_id}
                step_res = requests.post(f"{env_endpoint}/step", json=step_payload, timeout=TIMEOUT)
                step_res.raise_for_status()
                step_data = step_res.json()
                step_block = step_data["result"]

                raw_observation = step_block.get("observation", "")
                formatted_observation = format_observation(raw_observation)
                step_reward = step_block.get("reward", 0)
                done = step_block.get("done", False)

            except Exception as e:
                print(f"Step failed: {e}")
                step_reward = -0.01
                done = False

            turn_number += 1

        if done:
            print(
                f"[GT] Game {game_id} ended during strategy forcing phase at turn {turn_number}. "
                f"Returning fallback."
            )
            return index, None

        messages.append({"role": "user", "content": formatted_observation})
        expected_optimal_action = compute_optimal_action(formatted_observation)

        with rollout_last_prompt_and_completion_parallelized_curriculum.generation_semaphore:
            rollout_out = generate_rollout_completions(
                trainer, prompts=[messages], as_chat=True
            )[0]

        prompt_ids = rollout_out.get("prompt_ids", [])
        completion_ids = rollout_out.get("completion_ids", [])
        logprobs = rollout_out.get("logprobs", [])
        completion_text = tokenizer.decode(
            completion_ids, skip_special_tokens=True
        ).strip()

        messages.append({"role": "assistant", "content": completion_text})

        # Parse action from model output
        action_to_send = remove_reasoning_tags(completion_text)
        if action_to_send.endswith("</s>"):
            action_to_send = action_to_send[:-5]
        if "Action:" in action_to_send:
            action_to_send = action_to_send.split("Action:")[-1].strip()

        # Check strategy adherence for training turn
        strategy_followed = False
        try:
            model_action = int(action_to_send.strip())
            if expected_optimal_action is not None:
                strategy_followed = (model_action == expected_optimal_action)
        except Exception:
            pass

        # Check for invalid action
        invalid_action = False
        try:
            action_id_parsed = int(action_to_send.strip())
            legal_actions = parse_legal_actions(formatted_observation)
            if legal_actions and action_id_parsed not in legal_actions:
                print(f"Invalid action: {action_id_parsed} not in legal actions: {legal_actions}")
                invalid_action = True
        except Exception:
            invalid_action = True
            print(f"Invalid action: {action_to_send}")

        if invalid_action:
            print(f"Messages: {messages}")
            reward = INVALID_PENALTY
        elif strategy_followed:
            response_length = len(completion_ids)
            prompt_length = len(prompt_ids)
            len_reward_scale = max(0.2, min(5, prompt_length / response_length))
            reward = STRATEGY_REWARD * len_reward_scale
        else:
            reward = 0.0

        print("--------------------------------")
        print(
            f"[GT] game={game_id} train_turn={target_training_turn} "
            f"strategy={strategy_followed} "
            f"reward={reward:.3f} hints={use_hints}"
        )
        print("--------------------------------")

        return index, {
            "prompt_ids": prompt_ids,
            "completion_ids": completion_ids,
            "logprobs": logprobs,
            "reward": reward,
            "strategy_followed": strategy_followed,
        }

    # Execute episodes in parallel
    results = [None] * len(prompts)
    executor = rollout_last_prompt_and_completion_parallelized_curriculum.thread_pool

    futures = [
        executor.submit(run_single_prompt, i, p) for i, p in enumerate(prompts)
    ]

    for f in as_completed(futures):
        idx, res = f.result()
        if res is not None:
            results[idx] = res
        else:
            results[idx] = {
                "prompt_ids": [1],
                "completion_ids": [1],
                "logprobs": [1.0],
                "reward": 0.0,
                "strategy_followed": False,
            }

    # Update curriculum
    curriculum.step(len(prompts))

    # Log batch stats
    valid = [r for r in results if r is not None]
    if valid:
        avg_strat = sum(1 for r in valid if r["strategy_followed"]) / len(valid)
        avg_reward = sum(r["reward"] for r in valid) / len(valid)
        print(
            f"[GT-BATCH] Strategy: {avg_strat:.1%}, Avg Reward: {avg_reward:.3f}"
        )

    return {
        "prompt_ids": [r["prompt_ids"] for r in results],
        "completion_ids": [r["completion_ids"] for r in results],
        "logprobs": [r["logprobs"] for r in results],
        "env_rewards": [r["reward"] for r in results],
    }


# ============================================================
# Rollout Function 3: Full episode with strategy tracking
# ============================================================
def rollout_full_prompt_and_completion_parallelized_curriculum(
    prompts: list[str],
    trainer,
    max_turns: int = 30,
) -> dict[str, list]:
    """
    Parallelized rollout function for Gin Rummy.
    Uses full prompt and completion IDs with action masking.
    """
    # --- Constants for context length management ---
    MAX_EPISODE_TOKENS = 16384
    MAX_PROMPT_LEN = 4225

    # --- Reward Shaping Parameters ---
    STRATEGY_REWARD_WEIGHT = 0.5
    STEP_STRATEGY_REWARD = 0.1

    games_to_task_id_range = {
        "goofspiel": (0, 99999999),
        "liars_dice": (100000000, 199999999),
        "leduc_poker": (200000000, 299999999),
        "gin_rummy": (300000000, 399999999),
        "othello": (400000000, 499999999),
        "backgammon": (500000000, 599999999),
        "hex": (600000000, 699999999),
        "clobber": (700000000, 799999999),
    }

    selected_game = "gin_rummy"

    # --- 1. Static Initialization (Once per Rank) ---
    if not getattr(rollout_full_prompt_and_completion_parallelized_curriculum, "initialized", False):
        rank = int(os.environ.get("LOCAL_RANK", "0"))
        raw_urls = os.environ.get("ENVIRONMENT_SERVER_URLS", "")
        server_urls = [u.strip() for u in raw_urls.split(",") if u.strip()]

        if not server_urls:
            raise RuntimeError("ENVIRONMENT_SERVER_URLS is empty")

        env_pool = []

        for idx, base_url in enumerate(server_urls):
            try:
                print(f"[INIT] Initializing env on server {idx}: {base_url}")
                payload = {"task_id": games_to_task_id_range[selected_game][0], "seed": 42, "opponent": "mcts"}
                res = requests.post(f"{base_url}/reset", json=payload, timeout=300)
                res.raise_for_status()
                env_pool.append({"base_url": base_url})
                print(f"[INIT] Server {idx} ready")
            except Exception as e:
                raise RuntimeError(f"Failed to init server {base_url}: {e}")

        rollout_full_prompt_and_completion_parallelized_curriculum.rank = rank
        rollout_full_prompt_and_completion_parallelized_curriculum.env_pool = env_pool
        rollout_full_prompt_and_completion_parallelized_curriculum.num_servers = len(env_pool)
        rollout_full_prompt_and_completion_parallelized_curriculum.initialized = True
        rollout_full_prompt_and_completion_parallelized_curriculum.thread_pool = ThreadPoolExecutor(max_workers=len(env_pool))
        rollout_full_prompt_and_completion_parallelized_curriculum.generation_semaphore = Semaphore(1)
        rollout_full_prompt_and_completion_parallelized_curriculum.games_to_task_id_range = games_to_task_id_range
        rollout_full_prompt_and_completion_parallelized_curriculum.selected_game = selected_game

        # Initialize curriculum scheduler
        rollout_full_prompt_and_completion_parallelized_curriculum.curriculum = CurriculumScheduler(
            initial_max_turn=trainer.args.initial_max_turn,
            final_max_turn=50,
            rollouts_per_stage=trainer.args.rollouts_per_stage,
            initial_hint_prob=0.75,
            final_hint_prob=0.0,
            warmup_rollouts=trainer.args.rollouts_per_stage,
        )
        print(f"[CURRICULUM] Initialized with initial_max_turn={trainer.args.initial_max_turn}, final_max_turn=50, rollouts_per_stage={trainer.args.rollouts_per_stage}, initial_hint_prob=0.75, final_hint_prob=0.0, warmup_rollouts={trainer.args.rollouts_per_stage}")

    # Retrieve static variables
    rank = rollout_full_prompt_and_completion_parallelized_curriculum.rank
    env_pool = rollout_full_prompt_and_completion_parallelized_curriculum.env_pool
    num_servers = rollout_full_prompt_and_completion_parallelized_curriculum.num_servers
    games_to_task_id_range = rollout_full_prompt_and_completion_parallelized_curriculum.games_to_task_id_range
    selected_game = rollout_full_prompt_and_completion_parallelized_curriculum.selected_game
    curriculum = rollout_full_prompt_and_completion_parallelized_curriculum.curriculum

    tokenizer = trainer.processing_class
    TIMEOUT = 2400

    # Get current curriculum parameters
    total_rollouts = curriculum.total_rollouts
    current_max_turn = curriculum.get_max_turn()
    current_hint_prob = curriculum.get_hint_prob()
    print(f"[CURRICULUM] Rollout {total_rollouts}: max_turn={current_max_turn}, hint_prob={current_hint_prob:.2f}")

    def run_single_prompt(index: int, prompt: str):
        game_id = int(prompt)

        server_idx = (index + rank) % num_servers
        server = env_pool[server_idx]
        env_endpoint = server["base_url"]

        episode_prompt_ids: list[int] = []
        episode_completion_ids: list[int] = []
        episode_logprobs: list[float] = []
        episode_action_mask: list[int] = []
        prev_full_ids: list[int] | None = None
        invalid_count = 0
        done = False
        train_reward = 0.0
        turn_number = 0

        # Track strategy adherence
        strategy_followed_count = 0
        total_strategy_opportunities = 0
        step_rewards = []
        all_steps_correct = True
        # Determine if this episode gets hints
        use_hints = random.random() < current_hint_prob

        # --- Reset Environment (POST /reset) ---
        payload = {"task_id": game_id, "seed": 42, "opponent": "mcts"}

        try:
            reset_res = requests.post(f"{env_endpoint}/reset", json=payload, timeout=TIMEOUT)
            reset_res.raise_for_status()
            reset_data = reset_res.json()
            result_block = reset_data["result"]

            episode_id = result_block.get("episode_id", "")

            raw_observation = result_block.get("observation", "")
            formatted_observation = format_observation(raw_observation)

        except Exception as e:
            print(f"Failed to reset environment (Game {game_id}): {e}")
            return index, None

        # --- Build Conversation History ---
        system_prompt = GIN_RUMMY_SYSTEM_PROMPT

        if use_hints:
            system_prompt += GIN_RUMMY_HINT

        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": formatted_observation}]

        # --- Interaction Loop ---
        while not done and (turn_number < current_max_turn):
            # Compute optimal action for this observation
            expected_optimal = compute_optimal_action(formatted_observation)

            if turn_number == 0:
                print(f"[DEBUG-OBS] game={game_id} turn=0 observation='{formatted_observation[:500]}'")
                print(f"[DEBUG-OBS] legal_actions={parse_legal_actions(formatted_observation)} expected={expected_optimal}")

            # Generate Rollout Completion
            with rollout_full_prompt_and_completion_parallelized_curriculum.generation_semaphore:
                rollout_outputs = generate_rollout_completions(trainer, prompts=[messages], as_chat=True)[0]

            prompt_ids = rollout_outputs.get("prompt_ids", [])
            completion_ids = rollout_outputs.get("completion_ids", [])
            logprobs = rollout_outputs.get("logprobs", [])
            completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()

            # Check if prompt exceeds max length
            if len(prompt_ids) > MAX_PROMPT_LEN:
                print(f"Warning: Prompt exceeded {MAX_PROMPT_LEN} tokens ({len(prompt_ids)}) at turn {turn_number}, ending episode early")
                done = True
                break

            if turn_number == 0:
                episode_prompt_ids = prompt_ids
                prev_full_ids = prompt_ids.copy()
            else:
                if prev_full_ids is None:
                    prev_full_ids = prompt_ids.copy()
                elif prompt_ids[: len(prev_full_ids)] != prev_full_ids:
                    print(
                        f"Warning: BPE mismatch at turn {turn_number} (expected prefix {len(prev_full_ids)}, "
                        f"got {len(prompt_ids)} tokens). Skipping delta mask for this turn."
                    )
                    prev_full_ids = prompt_ids.copy()
                else:
                    delta_prompt_ids = prompt_ids[len(prev_full_ids):]
                    if delta_prompt_ids:
                        episode_completion_ids.extend(delta_prompt_ids)
                        episode_logprobs.extend([0.0] * len(delta_prompt_ids))
                        episode_action_mask.extend([0] * len(delta_prompt_ids))
                    prev_full_ids = prompt_ids.copy()

            if completion_ids:
                episode_completion_ids.extend(completion_ids)
                episode_logprobs.extend(logprobs)
                episode_action_mask.extend([1] * len(completion_ids))
                if prev_full_ids is not None:
                    prev_full_ids = prev_full_ids + completion_ids
            messages.append({"role": "assistant", "content": completion_text})

            # --- Parse Action ---
            action_to_send = remove_reasoning_tags(completion_text)
            if action_to_send.endswith("</s>"):
                action_to_send = action_to_send[:-4]

            if "Action:" in action_to_send:
                action_to_send = action_to_send.split("Action:")[-1].strip()

            # Robust number extraction: find last integer in the string
            number_match = re.findall(r'\b(\d+)\b', action_to_send)
            if number_match:
                action_to_send = number_match[-1]

            # DEBUG: log what the model generates vs what we expect
            if turn_number < 3:
                print(f"[DEBUG] game={game_id} turn={turn_number} raw='{completion_text[:120]}' parsed='{action_to_send}' expected={expected_optimal} legal={parse_legal_actions(formatted_observation)[:10]}")

            # --- Check Strategy Adherence ---
            try:
                model_action = int(action_to_send.strip())
                total_strategy_opportunities += 1
                if expected_optimal is not None and model_action == expected_optimal and all_steps_correct:
                    strategy_followed_count += 1
                    step_rewards.append(STEP_STRATEGY_REWARD)
                else:
                    all_steps_correct = False
                    step_rewards.append(0.0)
            except Exception:
                total_strategy_opportunities += 1
                all_steps_correct = False
                step_rewards.append(0.0)
                print(f"[DEBUG-FAIL] game={game_id} turn={turn_number} could not parse int from: '{action_to_send}'")

            # --- Step Environment (POST /step) ---
            try:
                formatted_observation = ""
                step_payload = {"action": action_to_send, "episode_id": episode_id}
                step_res = requests.post(f"{env_endpoint}/step", json=step_payload, timeout=TIMEOUT)
                step_res.raise_for_status()
                step_data = step_res.json()
                step_block = step_data["result"]

                raw_observation = step_block.get("observation", "")
                formatted_observation = format_observation(raw_observation)
                step_reward = step_block.get("reward", 0)
                done = step_block.get("done", False)

            except Exception as e:
                print(f"Step failed: {e}")
                step_reward = -0.01
                done = False
                invalid_count += 1

            # Check for invalid actions in observation
            if "Nothing happens" in formatted_observation or "Invalid" in formatted_observation:
                invalid_count += 1

            if done:
                train_reward = step_reward
            else:
                messages.append({"role": "user", "content": formatted_observation})

            turn_number += 1

        # Truncate episode if completion sequence exceeds max length
        if len(episode_completion_ids) > MAX_EPISODE_TOKENS:
            print(f"Warning: Episode completion exceeded {MAX_EPISODE_TOKENS} tokens ({len(episode_completion_ids)}), truncating")
            episode_completion_ids = episode_completion_ids[:MAX_EPISODE_TOKENS]
            episode_logprobs = episode_logprobs[:MAX_EPISODE_TOKENS]
            episode_action_mask = episode_action_mask[:MAX_EPISODE_TOKENS]

        # --- Calculate Final Reward with Strategy Shaping ---
        strategy_ratio = strategy_followed_count / total_strategy_opportunities if total_strategy_opportunities > 0 else 0.0

        immediate_rewards = sum(step_rewards)

        if not done:
            shaped_reward = immediate_rewards + strategy_ratio
        else:
            shaped_reward = (
                STRATEGY_REWARD_WEIGHT * strategy_ratio +
                (1 - STRATEGY_REWARD_WEIGHT) * train_reward +
                immediate_rewards
            )

        shaped_reward = shaped_reward - 0.05 * float(invalid_count)

        print("============")
        print(f"id: {game_id}, max_turn: {current_max_turn}, hints: {use_hints}", f"Strategy: {strategy_followed_count}/{total_strategy_opportunities} ({strategy_ratio:.2%})")
        print("============")

        return index, {
            "prompt_ids": episode_prompt_ids,
            "completion_ids": episode_completion_ids,
            "action_mask": episode_action_mask,
            "logprobs": episode_logprobs,
            "reward": shaped_reward,
            "strategy_ratio": strategy_ratio,
            "final_score": train_reward,
        }

    # --- Execute in parallel ---
    results = [None] * len(prompts)
    executor = rollout_full_prompt_and_completion_parallelized_curriculum.thread_pool

    futures = [
        executor.submit(run_single_prompt, i, p)
        for i, p in enumerate(prompts)
    ]

    for f in as_completed(futures):
        idx, res = f.result()
        if res is not None:
            results[idx] = res
        else:
            results[idx] = {
                "prompt_ids": [1],
                "completion_ids": [1],
                "action_mask": [0],
                "logprobs": [1.0],
                "reward": 0.0,
                "strategy_ratio": 0.0,
                "final_score": 0.0,
            }

    # Update curriculum after batch
    curriculum.step(len(prompts))

    list_results = [r for r in results if r is not None]

    avg_strategy = sum(r["strategy_ratio"] for r in list_results) / len(list_results) if list_results else 0
    avg_final = sum(r["final_score"] for r in list_results) / len(list_results) if list_results else 0
    print(f"[BATCH] Avg Strategy Adherence: {avg_strategy:.2%}, Avg Final Score: {avg_final:.3f}")

    return {
        "prompt_ids": [r["prompt_ids"] for r in list_results],
        "completion_ids": [r["completion_ids"] for r in list_results],
        "action_mask": [r["action_mask"] for r in list_results],
        "logprobs": [r["logprobs"] for r in list_results],
        "env_rewards": [r["reward"] for r in list_results],
    }


def rollout_reward_func(completions, **kwargs):
    rewards = kwargs.get("env_rewards") if kwargs else None
    return [float(r) for r in rewards] if rewards is not None else [0.0] * len(completions)
