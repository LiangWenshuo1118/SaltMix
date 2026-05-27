from __future__ import annotations

import argparse
from pathlib import Path

PROPERTY_CHOICES = ["density", "viscosity", "surface_tension", "electrical_conductivity"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Property-driven RK mixer workflow")
    subparsers = parser.add_subparsers(dest="command", required=True)

    preprocess_parser = subparsers.add_parser("preprocess")
    preprocess_parser.add_argument("--config", required=True)
    preprocess_parser.add_argument("--property", required=True, choices=PROPERTY_CHOICES)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--config", required=True)
    train_parser.add_argument("--property", required=True, choices=PROPERTY_CHOICES)

    eval_parser = subparsers.add_parser("eval")
    eval_parser.add_argument("--config", required=True)
    eval_parser.add_argument("--property", required=True, choices=PROPERTY_CHOICES)
    eval_parser.add_argument("--checkpoint", required=False)

    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("--config", required=True)
    predict_parser.add_argument("--property", required=True, choices=PROPERTY_CHOICES)
    predict_parser.add_argument("--input", required=True)
    predict_parser.add_argument("--output", required=True)
    predict_parser.add_argument("--checkpoint", required=False)
    predict_parser.add_argument("--coverage-output", required=False)

    args = parser.parse_args()
    config_path = Path(args.config)

    if args.command == "preprocess":
        from .preprocess import preprocess_from_config

        summary = preprocess_from_config(config_path, args.property)
        print(summary)
        return
    if args.command == "train":
        from .train import train_from_config

        summary = train_from_config(config_path, args.property)
        print(summary)
        return
    if args.command == "eval":
        from .train import evaluate_from_config

        checkpoint = Path(args.checkpoint) if args.checkpoint else None
        summary = evaluate_from_config(config_path, args.property, checkpoint)
        print(summary)
        return
    if args.command == "predict":
        from .inference import predict_from_config

        checkpoint = Path(args.checkpoint) if args.checkpoint else None
        coverage_output = Path(args.coverage_output) if args.coverage_output else None
        summary = predict_from_config(
            config_path,
            args.property,
            Path(args.input),
            Path(args.output),
            checkpoint,
            coverage_output,
        )
        print(summary)
        return
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
