"""Poster figures from the screenshots of capture_frames.py: crops of the chosen frames into fig/.

    python make_figures.py          # from this folder, with super_smash/.venv (needs Pillow)

frames/<session>/ holds the raw captures (1280x923 Dolphin window, not in git); the choices below say
which frame and which part of it each figure uses (left, top, right, bottom in pixels).
fig/kepiti_hero.jpg comes from Kepiti's poster (his imitation + RL model ending a combo with a shine).
"""

import os

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
FRAMES, FIG = os.path.join(HERE, "frames"), os.path.join(HERE, "fig")
THUMB = (560, 420)  # 4:3

# One thumbnail per agent: (image, crop box).
AGENTS = {
    "agent_rl.jpg": ("frames/mk5/0022_f03087.png", (130, 260, 690, 680)),  # back-air on the CPU at 33%
    "agent_delay.jpg": ("frames/mk9/0013_f03182.png", (300, 330, 860, 750)),  # dashing in on the CPU
    "agent_il.jpg": ("frames/il/0011_f01452.png", (80, 300, 640, 720)),  # a hit at 25%
    "agent_ilrl.jpg": ("fig/kepiti_hero.jpg", (213, 0, 906, 520)),  # shine at the end of a combo
}
# The hero strip: what pure RL learned against the level 9 CPU, left to right.
STRIP = [
    ("frames/mk5/0002_f00541.png", (120, 230, 1190, 880)),  # lasers from across the stage: 98%
    ("frames/mk5/0004_f00860.png", (230, 230, 1096, 880)),  # closes in at 126%
    ("frames/mk5/0016_f02255.png", (330, 230, 1196, 880)),  # the knockout
]
GAP = 14  # white pixels between strip panels


def main():
    os.makedirs(FIG, exist_ok=True)
    for name, (source, box) in AGENTS.items():
        image = Image.open(os.path.join(HERE, source)).convert("RGB").crop(box).resize(THUMB, Image.LANCZOS)
        image.save(os.path.join(FIG, name), quality=92)
        print(f"{name}: {source} {box}")
    panels = [Image.open(os.path.join(HERE, source)).convert("RGB").crop(box) for source, box in STRIP]
    height = min(p.size[1] for p in panels)
    panels = [p.resize((round(p.size[0] * height / p.size[1]), height), Image.LANCZOS) for p in panels]
    strip = Image.new("RGB", (sum(p.size[0] for p in panels) + GAP * (len(panels) - 1), height), "white")
    x = 0
    for p in panels:
        strip.paste(p, (x, 0))
        x += p.size[0] + GAP
    strip.save(os.path.join(FIG, "hero.jpg"), quality=92)
    print(f"hero.jpg: {len(panels)} panels, {strip.size[0]}x{strip.size[1]}")


if __name__ == "__main__":
    main()
