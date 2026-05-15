import argparse
import subprocess
import sys
import yaml
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Run all NeRF ablation experiments defined in a YAML config')
    parser.add_argument('--config',   type=str, default='configs/nerf_ablation_runs.yml')
    parser.add_argument('--no_wandb',   action='store_true')
    parser.add_argument('--use_kernel', action='store_true', help='Enable fused CUDA PE kernel')
    parser.add_argument('--compile',    action='store_true', help='Enable torch.compile on the model')
    parser.add_argument('--dry_run',    action='store_true', help='Print commands without running them')
    args = parser.parse_args()

    cfg      = yaml.safe_load(Path(args.config).read_text())
    defaults = cfg.get('defaults', {})
    runs     = cfg['runs']

    print(f"Loaded {len(runs)} runs from {args.config}\n")

    for i, run in enumerate(runs):
        params   = {**defaults, **run}
        arch_tag = f"n{params['hidden_layers']}{params['hidden_features']}_f{params['L_freq']}"

        cmd = [
            sys.executable, '-m', 'src.nerf',
            '--input',                   str(params['input']),
            '--model_dir',               str(params['model_dir']),
            '--hidden_layers',           str(params['hidden_layers']),
            '--hidden_features',         str(params['hidden_features']),
            '--L_freq',                  str(params['L_freq']),
            '--skip_at',                 str(params['skip_at']),
            '--lr',                      str(params['lr']),
            '--epochs',                  str(params['epochs']),
            '--batch_size',              str(params['batch_size']),
            '--steps_til_summary',       str(params['steps_til_summary']),
            '--epochs_til_checkpoint',   str(params['epochs_til_checkpoint']),
            '--early_stopping_patience', str(params['early_stopping_patience']),
            '--wandb_project',           str(params['wandb_project']),
            '--resume',
        ]
        if args.use_kernel:
            cmd.append('--use_kernel')
        if args.compile:
            cmd.append('--compile')
        if args.no_wandb:
            cmd.append('--no_wandb')

        report_path = Path(params['model_dir']) / arch_tag / 'report.json'

        print(f"[{i+1}/{len(runs)}] {arch_tag}")

        if report_path.exists():
            print(f"  Skipping — report.json already exists\n")
            continue

        log_path = Path(params['model_dir']) / arch_tag / 'train.log'
        log_path.parent.mkdir(parents=True, exist_ok=True)

        print('  ' + ' '.join(cmd))
        print(f'  log → {log_path}')

        if args.dry_run:
            print('  (dry run — skipped)\n')
            continue

        cmd_tee = ' '.join(cmd) + f' 2>&1 | tee {log_path}'
        result = subprocess.run(cmd_tee, shell=True)
        if result.returncode != 0:
            print(f"\nRun {arch_tag} failed (exit {result.returncode}). Stopping.")
            sys.exit(result.returncode)

        print(f"  Done: {arch_tag}\n")

    print("All runs complete.")


if __name__ == '__main__':
    main()
