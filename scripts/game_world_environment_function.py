import os
import re
import random
import requests
from itertools import combinations
from concurrent.futures import ThreadPoolExecutor, as_completed
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
            self.initial_max_turn + stage * 2,
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
# Rollout Function 2: Lockstep batched single-turn rollout
# ============================================================
def rollout_last_prompt_and_completion_parallelized_curriculum(
    prompts: list[str],
    trainer,
    max_turns: int = 30,
) -> dict[str, list]:
    """
    Lockstep batched Rollout 2 for Gin Rummy with privileged learning.

    4-phase approach (no Semaphore, single vLLM call):
      Phase 1: Parallel env resets (ThreadPoolExecutor)
      Phase 2: Parallel strategy forcing (ThreadPoolExecutor)
               Each thread plays turns 0..target_turn-1 using compute_optimal_action()
      Phase 3: Single batched generation for all active episodes
      Phase 4: Evaluate rewards (optimal=1.0, legal=0.1, invalid=-0.1)

    Teacher/student split: even index = teacher (hint based on hint_prob),
                           odd index = student (never hint).
    """
    # --- Constants ---
    STRATEGY_REWARD = 1.0
    LEGAL_REWARD = 0.1
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
    fn = rollout_last_prompt_and_completion_parallelized_curriculum

    # --- 1. Static Initialization (Once per Rank) ---
    if not getattr(fn, "initialized", False):
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

        fn.rank = rank
        fn.env_pool = env_pool
        fn.num_servers = len(env_pool)
        fn.initialized = True
        fn.thread_pool = ThreadPoolExecutor(max_workers=max(16, len(env_pool)))

        # Curriculum: fast progression to max_turn=30
        fn.curriculum = CurriculumScheduler(
            initial_max_turn=trainer.args.initial_max_turn,
            final_max_turn=30,
            rollouts_per_stage=96,
            initial_hint_prob=0.75,
            final_hint_prob=0.0,
            warmup_rollouts=96,
        )
        print(f"[CURRICULUM] init max_turn={trainer.args.initial_max_turn}->30, stage=96, hint=0.75->0.0")

    # Retrieve static variables
    rank = fn.rank
    env_pool = fn.env_pool
    num_servers = fn.num_servers
    curriculum = fn.curriculum
    thread_pool = fn.thread_pool

    tokenizer = trainer.processing_class
    TIMEOUT = 2400
    num_episodes = len(prompts)

    # Get current curriculum parameters
    current_max_turn = curriculum.get_max_turn()
    current_hint_prob = curriculum.get_hint_prob()
    target_turn = current_max_turn - 1
    print(f"[CURRICULUM] Rollout {curriculum.total_rollouts}: max_turn={current_max_turn}, hint_prob={current_hint_prob:.2f}")

    # ========================================
    # Phase 1: Parallel env resets
    # ========================================
    episodes = [None] * num_episodes

    def reset_env(index):
        game_id = int(prompts[index])
        server_idx = (index + rank) % num_servers
        env_endpoint = env_pool[server_idx]["base_url"]
        payload = {"task_id": game_id, "seed": 42, "opponent": "mcts"}
        try:
            res = requests.post(f"{env_endpoint}/reset", json=payload, timeout=TIMEOUT)
            res.raise_for_status()
            result = res.json()["result"]
            return index, {
                "game_id": game_id,
                "episode_id": result.get("episode_id", ""),
                "observation": format_observation(result.get("observation", "")),
                "env_endpoint": env_endpoint,
                "failed": False,
            }
        except Exception as e:
            print(f"Reset failed for game {game_id}: {e}")
            return index, {
                "game_id": game_id,
                "episode_id": "",
                "observation": "",
                "env_endpoint": env_endpoint,
                "failed": True,
            }

    reset_futures = [thread_pool.submit(reset_env, i) for i in range(num_episodes)]
    for f in as_completed(reset_futures):
        idx, data = f.result()
        episodes[idx] = data

    # Set teacher/student hints
    for i in range(num_episodes):
        ep = episodes[i]
        is_teacher = (i % 2 == 0)
        ep["is_teacher"] = is_teacher
        ep["use_hints"] = is_teacher and (random.random() < current_hint_prob)

    # ========================================
    # Phase 2: Parallel strategy forcing
    # ========================================
    def force_episode(index):
        """Play turns 0..target_turn-1 using optimal actions. Returns (index, messages, observation, done)."""
        ep = episodes[index]
        if ep["failed"]:
            return index, None, None, True

        messages = [{"role": "system", "content": GIN_RUMMY_SYSTEM_PROMPT}]
        obs = ep["observation"]
        episode_id = ep["episode_id"]
        env_endpoint = ep["env_endpoint"]
        done = False

        for turn in range(target_turn):
            messages.append({"role": "user", "content": obs})

            optimal_action = compute_optimal_action(obs)
            if optimal_action is None:
                break

            messages.append({"role": "assistant", "content": str(optimal_action)})

            try:
                step_payload = {"action": str(optimal_action), "episode_id": episode_id}
                step_res = requests.post(f"{env_endpoint}/step", json=step_payload, timeout=TIMEOUT)
                step_res.raise_for_status()
                step_block = step_res.json()["result"]
                obs = format_observation(step_block.get("observation", ""))
                done = step_block.get("done", False)
            except Exception as e:
                print(f"Strategy forcing step failed for episode {index}: {e}")
                done = True
                break

            if done:
                break

        return index, messages, obs, done

    forcing_results = [None] * num_episodes
    force_futures = [thread_pool.submit(force_episode, i) for i in range(num_episodes)]
    for f in as_completed(force_futures):
        idx, messages, obs, done = f.result()
        forcing_results[idx] = (messages, obs, done)

    # ========================================
    # Phase 3: Single batched generation
    # ========================================
    # Collect prompts from active (non-done) episodes
    active_indices = []
    active_prompts = []
    active_expected = []

    for i in range(num_episodes):
        messages, obs, done = forcing_results[i]
        if done or messages is None:
            continue

        expected_optimal = compute_optimal_action(obs)
        obs_for_model = obs
        if episodes[i]["use_hints"] and expected_optimal is not None:
            obs_for_model += f"\n\n[HINT: The best action here is {expected_optimal}]"
        messages.append({"role": "user", "content": obs_for_model})

        active_indices.append(i)
        active_prompts.append(messages)
        active_expected.append(expected_optimal)
        # Store observation for reward evaluation
        episodes[i]["_final_obs"] = obs

    # Single vLLM call for ALL active episodes
    gen_outputs = {}
    if active_prompts:
        all_outputs = generate_rollout_completions(
            trainer, prompts=active_prompts, as_chat=True
        )
        for j, i in enumerate(active_indices):
            gen_outputs[i] = (all_outputs[j], active_expected[j])

    # ========================================
    # Phase 4: Evaluate rewards & build results
    # ========================================
    results = [None] * num_episodes

    for i in range(num_episodes):
        ep = episodes[i]

        if i not in gen_outputs:
            # Episode ended during forcing or failed reset — fallback
            results[i] = {
                "prompt_ids": [1],
                "completion_ids": [1],
                "logprobs": [1.0],
                "reward": 0.0,
                "strategy_followed": False,
            }
            continue

        output, expected_optimal = gen_outputs[i]
        prompt_ids = output.get("prompt_ids", [])
        completion_ids = output.get("completion_ids", [])
        logprobs = output.get("logprobs", [])
        completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()

        # Parse action
        action_text = remove_reasoning_tags(completion_text)
        if action_text.endswith("</s>"):
            action_text = action_text[:-4]
        if "Action:" in action_text:
            action_text = action_text.split("Action:")[-1].strip()
        numbers = re.findall(r'\b(\d+)\b', action_text)
        if numbers:
            action_text = numbers[-1]

        # Evaluate
        strategy_followed = False
        invalid_action = False
        try:
            model_action = int(action_text.strip())
            legal_actions = parse_legal_actions(ep["_final_obs"])
            if legal_actions and model_action not in legal_actions:
                invalid_action = True
            if expected_optimal is not None:
                strategy_followed = (model_action == expected_optimal)
        except Exception:
            invalid_action = True

        # Compute reward
        if invalid_action:
            reward = INVALID_PENALTY
        elif strategy_followed:
            reward = STRATEGY_REWARD
        else:
            reward = LEGAL_REWARD

        is_teacher = ep["is_teacher"]
        print(
            f"[GT] game={ep['game_id']} turn={target_turn} "
            f"strat={strategy_followed} reward={reward:.2f} "
            f"hint={ep['use_hints']} role={'teacher' if is_teacher else 'student'}"
        )

        results[i] = {
            "prompt_ids": prompt_ids,
            "completion_ids": completion_ids,
            "logprobs": logprobs,
            "reward": reward,
            "strategy_followed": strategy_followed,
        }

    # Update curriculum
    curriculum.step(num_episodes)

    # Log batch stats
    teachers = [results[i] for i in range(num_episodes) if episodes[i]["is_teacher"]]
    students = [results[i] for i in range(num_episodes) if not episodes[i]["is_teacher"]]
    t_strat = sum(1 for r in teachers if r["strategy_followed"]) / max(len(teachers), 1)
    s_strat = sum(1 for r in students if r["strategy_followed"]) / max(len(students), 1)
    avg_reward = sum(r["reward"] for r in results) / max(len(results), 1)
    print(
        f"[BATCH] Teacher: {t_strat:.0%}, Student: {s_strat:.0%}, "
        f"Avg Reward: {avg_reward:.3f}"
    )

    return {
        "prompt_ids": [r["prompt_ids"] for r in results],
        "completion_ids": [r["completion_ids"] for r in results],
        "logprobs": [r["logprobs"] for r in results],
        "env_rewards": [r["reward"] for r in results],
    }


# ============================================================
# Rollout Function 3: Lockstep batched multi-turn rollout
# ============================================================
def rollout_full_prompt_and_completion_parallelized_curriculum(
    prompts: list[str],
    trainer,
    max_turns: int = 30,
) -> dict[str, list]:
    """
    Lockstep batched rollout for Gin Rummy with curriculum learning.
    - All episodes advance turn-by-turn in sync
    - Generations are batched into a SINGLE vLLM call per turn (no Semaphore)
    - Env HTTP calls are parallelized via ThreadPoolExecutor
    - Action masking: 1 for model-generated tokens, 0 for env/user tokens
    - Teacher/student hint split with curriculum fade-out
    """
    # --- Constants ---
    MAX_EPISODE_TOKENS = 16384
    MAX_PROMPT_LEN = 4225
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
    fn = rollout_full_prompt_and_completion_parallelized_curriculum

    # --- Static Initialization (Once per Rank) ---
    if not getattr(fn, "initialized", False):
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

        fn.rank = rank
        fn.env_pool = env_pool
        fn.num_servers = len(env_pool)
        fn.initialized = True
        fn.thread_pool = ThreadPoolExecutor(max_workers=max(16, len(env_pool)))

        fn.curriculum = CurriculumScheduler(
            initial_max_turn=trainer.args.initial_max_turn,
            final_max_turn=30,
            rollouts_per_stage=96,
            initial_hint_prob=0.75,
            final_hint_prob=0.0,
            warmup_rollouts=96,
        )
        print(f"[CURRICULUM] init max_turn={trainer.args.initial_max_turn}->30, stage=96, hint=0.75->0.0")

    # Retrieve static variables
    rank = fn.rank
    env_pool = fn.env_pool
    num_servers = fn.num_servers
    curriculum = fn.curriculum
    thread_pool = fn.thread_pool

    tokenizer = trainer.processing_class
    TIMEOUT = 2400
    num_episodes = len(prompts)

    # Get curriculum params
    current_max_turn = curriculum.get_max_turn()
    current_hint_prob = curriculum.get_hint_prob()
    print(f"[CURRICULUM] Rollout {curriculum.total_rollouts}: max_turn={current_max_turn}, hint_prob={current_hint_prob:.2f}")

    # ========================================
    # Phase 1: Reset all environments in parallel
    # ========================================
    episodes = [None] * num_episodes

    def reset_env(index):
        game_id = int(prompts[index])
        server_idx = (index + rank) % num_servers
        env_endpoint = env_pool[server_idx]["base_url"]
        payload = {"task_id": game_id, "seed": 42, "opponent": "mcts"}
        try:
            res = requests.post(f"{env_endpoint}/reset", json=payload, timeout=TIMEOUT)
            res.raise_for_status()
            result = res.json()["result"]
            return index, {
                "game_id": game_id,
                "episode_id": result.get("episode_id", ""),
                "observation": format_observation(result.get("observation", "")),
                "env_endpoint": env_endpoint,
            }
        except Exception as e:
            print(f"Reset failed for game {game_id}: {e}")
            return index, None

    reset_futures = [thread_pool.submit(reset_env, i) for i in range(num_episodes)]
    for f in as_completed(reset_futures):
        idx, data = f.result()
        episodes[idx] = data

    # Initialize per-episode state
    for i in range(num_episodes):
        if episodes[i] is None:
            episodes[i] = {
                "game_id": int(prompts[i]),
                "episode_id": "",
                "observation": "",
                "env_endpoint": "",
            }
            episodes[i]["_failed"] = True
        else:
            episodes[i]["_failed"] = False

        ep = episodes[i]
        ep["messages"] = [{"role": "system", "content": GIN_RUMMY_SYSTEM_PROMPT}]
        ep["prompt_ids_first"] = []
        ep["completion_ids"] = []
        ep["logprobs"] = []
        ep["action_mask"] = []
        ep["prev_full_ids"] = None
        ep["done"] = ep.get("_failed", False)
        ep["active"] = not ep.get("_failed", False)
        ep["strategy_count"] = 0
        ep["strategy_total"] = 0
        ep["step_rewards"] = []
        ep["invalid_count"] = 0
        ep["train_reward"] = 0.0

        # Teacher/student hint split
        is_teacher = (i % 2 == 0)
        ep["is_teacher"] = is_teacher
        ep["use_hints"] = is_teacher and (random.random() < current_hint_prob)

    # ========================================
    # Phase 2: Lockstep turn loop
    # ========================================
    for turn in range(current_max_turn):
        active_indices = [i for i in range(num_episodes) if episodes[i]["active"] and not episodes[i]["done"]]
        if not active_indices:
            break

        # 2a. Prepare prompts: append observation + optional hint
        for i in active_indices:
            ep = episodes[i]
            obs = ep["observation"]
            expected_optimal = compute_optimal_action(obs)
            ep["_expected_optimal"] = expected_optimal

            obs_for_model = obs
            if ep["use_hints"] and expected_optimal is not None:
                obs_for_model = obs + f"\n\n[HINT: The best action here is {expected_optimal}]"
            ep["messages"].append({"role": "user", "content": obs_for_model})

        # 2b. Batched generation - SINGLE vLLM call for all active episodes
        prompts_to_gen = [episodes[i]["messages"] for i in active_indices]
        all_outputs = generate_rollout_completions(trainer, prompts=prompts_to_gen, as_chat=True)

        # 2c. Process generation outputs: delta computation + action masking
        for j, i in enumerate(active_indices):
            ep = episodes[i]
            output = all_outputs[j]

            prompt_ids = output.get("prompt_ids", [])
            completion_ids = output.get("completion_ids", [])
            logprobs = output.get("logprobs", [])
            completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()

            # Check prompt length limit
            if len(prompt_ids) > MAX_PROMPT_LEN:
                print(f"Prompt exceeded {MAX_PROMPT_LEN} at turn {turn}, ending episode {i}")
                ep["active"] = False
                continue

            # Delta computation for action masking
            if turn == 0:
                ep["prompt_ids_first"] = prompt_ids
                ep["prev_full_ids"] = prompt_ids.copy()
            else:
                prev = ep["prev_full_ids"]
                if prev is None:
                    ep["prev_full_ids"] = prompt_ids.copy()
                elif len(prompt_ids) >= len(prev) and prompt_ids[:len(prev)] == prev:
                    delta = prompt_ids[len(prev):]
                    if delta:
                        ep["completion_ids"].extend(delta)
                        ep["logprobs"].extend([0.0] * len(delta))
                        ep["action_mask"].extend([0] * len(delta))
                    ep["prev_full_ids"] = prompt_ids.copy()
                else:
                    print(f"BPE mismatch at turn {turn}, episode {i}")
                    ep["prev_full_ids"] = prompt_ids.copy()

            if completion_ids:
                ep["completion_ids"].extend(completion_ids)
                ep["logprobs"].extend(logprobs)
                ep["action_mask"].extend([1] * len(completion_ids))
                if ep["prev_full_ids"] is not None:
                    ep["prev_full_ids"] = ep["prev_full_ids"] + completion_ids

            ep["messages"].append({"role": "assistant", "content": completion_text})

            # Parse action
            action_text = remove_reasoning_tags(completion_text)
            if action_text.endswith("</s>"):
                action_text = action_text[:-4]
            if "Action:" in action_text:
                action_text = action_text.split("Action:")[-1].strip()
            numbers = re.findall(r'\b(\d+)\b', action_text)
            if numbers:
                action_text = numbers[-1]
            ep["_parsed_action"] = action_text

            # Strategy adherence check
            try:
                model_action = int(action_text.strip())
                ep["strategy_total"] += 1
                expected = ep.get("_expected_optimal")
                if expected is not None and model_action == expected:
                    ep["strategy_count"] += 1
                    ep["step_rewards"].append(STEP_STRATEGY_REWARD)
                else:
                    legal = parse_legal_actions(ep["observation"])
                    if legal and model_action in legal:
                        ep["step_rewards"].append(STEP_STRATEGY_REWARD * 0.2)
                    else:
                        ep["step_rewards"].append(0.0)
            except Exception:
                ep["strategy_total"] += 1
                ep["step_rewards"].append(0.0)

        # 2d. Step all envs in parallel
        def step_env(index):
            ep = episodes[index]
            if not ep["active"] or ep["done"]:
                return index
            action = ep.get("_parsed_action", "")
            try:
                payload = {"action": action, "episode_id": ep["episode_id"]}
                res = requests.post(f"{ep['env_endpoint']}/step", json=payload, timeout=TIMEOUT)
                res.raise_for_status()
                result = res.json()["result"]
                ep["observation"] = format_observation(result.get("observation", ""))
                done = result.get("done", False)
                ep["done"] = done
                if "Nothing happens" in ep["observation"] or "Invalid" in ep["observation"]:
                    ep["invalid_count"] += 1
                if done:
                    ep["train_reward"] = result.get("reward", 0)
            except Exception as e:
                print(f"Step failed for episode {index}: {e}")
                ep["invalid_count"] += 1
            return index

        step_futures = [thread_pool.submit(step_env, i) for i in active_indices]
        for f in as_completed(step_futures):
            f.result()

    # ========================================
    # Phase 3: Build results
    # ========================================
    results = []
    for i in range(num_episodes):
        ep = episodes[i]
        comp_ids = ep["completion_ids"]
        lps = ep["logprobs"]
        am = ep["action_mask"]

        if not comp_ids:
            results.append({
                "prompt_ids": [1],
                "completion_ids": [1],
                "action_mask": [0],
                "logprobs": [1.0],
                "reward": 0.0,
                "strategy_ratio": 0.0,
                "final_score": 0.0,
            })
            continue

        # Truncate if needed
        if len(comp_ids) > MAX_EPISODE_TOKENS:
            comp_ids = comp_ids[:MAX_EPISODE_TOKENS]
            lps = lps[:MAX_EPISODE_TOKENS]
            am = am[:MAX_EPISODE_TOKENS]

        strategy_ratio = ep["strategy_count"] / ep["strategy_total"] if ep["strategy_total"] > 0 else 0.0
        immediate_rewards = sum(ep["step_rewards"])

        if not ep["done"]:
            shaped_reward = immediate_rewards + strategy_ratio
        else:
            shaped_reward = (
                STRATEGY_REWARD_WEIGHT * strategy_ratio +
                (1 - STRATEGY_REWARD_WEIGHT) * ep["train_reward"] +
                immediate_rewards
            )
        shaped_reward -= 0.05 * ep["invalid_count"]

        print(
            f"============\n"
            f"id: {ep['game_id']}, max_turn: {current_max_turn}, hints: {ep['use_hints']}, "
            f"role={'teacher' if ep['is_teacher'] else 'student'}, "
            f"Strategy: {ep['strategy_count']}/{ep['strategy_total']} ({strategy_ratio:.2%})\n"
            f"============"
        )

        results.append({
            "prompt_ids": ep["prompt_ids_first"],
            "completion_ids": comp_ids,
            "action_mask": am,
            "logprobs": lps,
            "reward": shaped_reward,
            "strategy_ratio": strategy_ratio,
            "final_score": ep["train_reward"],
        })

    # Update curriculum
    curriculum.step(num_episodes)

    # Batch stats
    teachers = [r for i, r in enumerate(results) if episodes[i].get("is_teacher", False)]
    students = [r for i, r in enumerate(results) if not episodes[i].get("is_teacher", True)]
    t_strat = sum(r["strategy_ratio"] for r in teachers) / max(len(teachers), 1)
    s_strat = sum(r["strategy_ratio"] for r in students) / max(len(students), 1)
    avg_reward = sum(r["reward"] for r in results) / max(len(results), 1)
    print(f"[BATCH] Teacher: {t_strat:.2%}, Student: {s_strat:.2%}, Avg Reward: {avg_reward:.3f}")

    return {
        "prompt_ids": [r["prompt_ids"] for r in results],
        "completion_ids": [r["completion_ids"] for r in results],
        "action_mask": [r["action_mask"] for r in results],
        "logprobs": [r["logprobs"] for r in results],
        "env_rewards": [r["reward"] for r in results],
    }


def rollout_reward_func(completions, **kwargs):
    rewards = kwargs.get("env_rewards") if kwargs else None
    return [float(r) for r in rewards] if rewards is not None else [0.0] * len(completions)
