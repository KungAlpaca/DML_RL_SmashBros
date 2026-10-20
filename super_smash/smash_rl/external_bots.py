"""Other teams' bots (act(gs, controller) / reset() interface) as opponents for our env and party_teams.py.

    ("kepiti", "models/kepiti/fox_rl_3h.pt")                  Kepiti's Fox (LSTM, 18-frame reaction delay); needs
                                                               his repo (KEPITI_REPO or ~/Documents/DML/super_smash_RL)
    ("ganondorf", "models/ganondorf/ganondorf_edge_v4.zip")   the Ganondorf agent (ganondorf_agent.py next to it)

make_bot(spec, port, opponent_port) builds one; specs are plain tuples so they can be sent to env worker
processes, which build their bot there.
"""

import importlib
import os
import sys

import melee

KEPITI_REPO = os.environ.get("KEPITI_REPO", os.path.expanduser("~/Documents/DML/super_smash_RL"))


def import_kepiti_foxbot(repo: str = KEPITI_REPO):
    """Kepiti's FoxBot class. His package is also called smash_rl, so it is imported with ours set
    aside and then filed under 'kepiti_smash_rl.*' (his modules bind everything they use at import)."""
    cached = sys.modules.get("kepiti_smash_rl.fox_bot")
    if cached is not None:
        return cached.FoxBot
    ours = {k: v for k, v in sys.modules.items() if k == "smash_rl" or k.startswith("smash_rl.")}
    for k in ours:
        del sys.modules[k]
    sys.path.insert(0, repo)
    try:
        fox_bot = importlib.import_module("smash_rl.fox_bot")
    finally:
        sys.path.remove(repo)
        theirs = {k: v for k, v in sys.modules.items() if k == "smash_rl" or k.startswith("smash_rl.")}
        for k, v in theirs.items():
            del sys.modules[k]
            sys.modules["kepiti_" + k] = v
        sys.modules.update(ours)
    return fox_bot.FoxBot


def bot_character(spec) -> str:
    return "GANONDORF" if spec[0] == "ganondorf" else "FOX"


def make_bot(spec, port: int, opponent_port: int, deterministic: bool = True, kepiti_delay: int = 18,
             torch_threads: int = None):
    kind, path = spec
    if kind == "kepiti":
        if torch_threads:
            import torch
            torch.set_num_threads(torch_threads)
        FoxBot = import_kepiti_foxbot()
        return FoxBot(path, port=port, opponent_port=opponent_port, delay=kepiti_delay, device="cpu")
    if kind == "ganondorf":
        folder = os.path.dirname(os.path.abspath(path))
        if folder not in sys.path:
            sys.path.insert(0, folder)
        agent_cls = importlib.import_module("ganondorf_agent").GanondorfAgent
        return agent_cls(path, port=port, opponent_port=opponent_port, deterministic=deterministic)
    raise ValueError(f"unknown bot kind {kind!r}")
