#!/bin/bash

set -e

OBJECTIVE_DIRECTORY="${1:-tuned-roberta-base-text+latex.MLM-objective}"
LOSS_TYPE="${2:-eval_loss}"

export LC_ALL=C

(
    grep -F '"'"$LOSS_TYPE"'"' "$OBJECTIVE_DIRECTORY"/checkpoint-*/trainer_state.json |
    sed -r -e 's/.*checkpoint-([0-9]*)\/.*"'"$LOSS_TYPE"'": /\1\t/' -e 's/,$//' |
    tee \
        >(sort -k  2n -k 1n | head -n 1 | sed -e 's/^/min:  /' -e 's/\t/ /') \
        >(sort -k 1rn -k 2n | head -n 1 | sed -e 's/^/last: /' -e 's/\t/ /') \
        >/dev/null
) | sort -k 1r
