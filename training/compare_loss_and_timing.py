import argparse
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ParsedLog:
    loss_by_step: dict[int, float]
    timing_by_step: dict[int, float]


_STEP_RE = re.compile(r"\bglobal_step:\s*(\d+)\b")
_LOSS_RE = re.compile(r"\breduced_train_loss:\s*([0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)\b")
_TIMING_RE = re.compile(r"\btrain_step_timing in s:\s*([0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)\b")


def parse_log(path: Path) -> ParsedLog:
    loss_by_step: dict[int, float] = {}
    timing_by_step: dict[int, float] = {}

    with path.open("r", errors="ignore") as f:
        for line in f:
            m_step = _STEP_RE.search(line)
            if not m_step:
                continue
            step = int(m_step.group(1))

            m_loss = _LOSS_RE.search(line)
            if m_loss and step not in loss_by_step:
                loss_by_step[step] = float(m_loss.group(1))

            m_timing = _TIMING_RE.search(line)
            if m_timing and step not in timing_by_step:
                timing_by_step[step] = float(m_timing.group(1))

    return ParsedLog(loss_by_step=loss_by_step, timing_by_step=timing_by_step)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--b300", default=str(Path("training") / "wangshaojie05_qwen2-7b_log-nemo-megatron-Qwen3-MoE-235B_cudnn_fa.out"))
    #parser.add_argument("--b300", default=str(Path("training") / "b300_100steps.log"))
    parser.add_argument("--h200", default=str(Path("training") / "h200_200steps.log"))
    parser.add_argument("--out", default=str(Path("training") / "loss_compare.png"))
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    b300 = parse_log(Path(args.b300))
    h200 = parse_log(Path(args.h200))

    common_steps = sorted(set(b300.loss_by_step.keys()) & set(h200.loss_by_step.keys()))
    if not common_steps:
        raise SystemExit("没有找到两个log都包含的global_step / reduced_train_loss")

    b300_losses = [b300.loss_by_step[s] for s in common_steps]
    h200_losses = [h200.loss_by_step[s] for s in common_steps]

    rel_errs = [abs(a - b) / abs(b) if b != 0 else float("nan") for a, b in zip(b300_losses, h200_losses)]
    avg_rel_err = mean([e for e in rel_errs if e == e])

    b300_timing_steps = [s for s in range(6, 100) if s in b300.timing_by_step]
    h200_timing_steps = [s for s in range(6, 100) if s in h200.timing_by_step]
    b300_timing_avg_6_99 = mean([b300.timing_by_step[s] for s in b300_timing_steps])
    h200_timing_avg_6_99 = mean([h200.timing_by_step[s] for s in h200_timing_steps])

    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.plot(common_steps, b300_losses, label="b300")
    plt.plot(common_steps, h200_losses, label="h200")
    plt.xlabel("global_step")
    plt.ylabel("reduced_train_loss")
    plt.title("Reduced train loss compare")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    if args.show:
        plt.show()

    print(f"common_steps: {len(common_steps)} (min={common_steps[0]}, max={common_steps[-1]})")
    print(f"avg_relative_error(b300 vs h200): {avg_rel_err}")
    print(f"b300_avg_train_step_timing_s(step6-99): {b300_timing_avg_6_99} (n={len(b300_timing_steps)})")
    print(f"h200_avg_train_step_timing_s(step6-99): {h200_timing_avg_6_99} (n={len(h200_timing_steps)})")


if __name__ == "__main__":
    main()
