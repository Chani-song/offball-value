#!/usr/bin/env bash
#SBATCH --job-name=offball-audit
#SBATCH --cpus-per-task=8
#SBATCH --mem=96g
#SBATCH --time=02:00:00
#SBATCH --partition=cpu
#SBATCH --output=logs/audit_%j.out
#SBATCH --error=logs/audit_%j.out
#
# Two questions raised by actually looking at the scenes, both answerable
# from builds that already exist.
#
# 1 Defender choice. Candidates are ranked from the runner's side only --
#   goal-side marking distance and pursuit cost -- and the top three are
#   always taken. A first pass found the runner is not even the nearest
#   attacker to the chosen defender in 62 % of picks, median 6.7 m further
#   than the man he is actually standing next to. This measures how much
#   each possible filter would cut, and whether it cuts the defenders the
#   labels say really reacted.
#
# 2 Runner-destination xT. Rank scenes by the threat of the ground the
#   runner moves into, rather than by what the attack is worth. Different
#   from vacated_xt, which scores what the DEFENDER gives up. Tested the
#   same way as every other candidate criterion, against the same labels.

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
module load python || module load anaconda3_cpu || true
. .venv-delta/bin/activate
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONPATH=src
export PYTHONUNBUFFERED=1

echo "##################################################################"
echo "# 1. 수비수 선정 감사"
echo "##################################################################"
python scripts/audit_defender_choice.py || echo "(실패)"

echo
echo "##################################################################"
echo "# 2. 러너가 들어가는 지역의 xT"
echo "##################################################################"
python scripts/test_runner_destination_xt.py || echo "(실패)"
