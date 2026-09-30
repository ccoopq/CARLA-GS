#!/bin/bash
scenes=("002")
for scene in "${scenes[@]}"; do
   python render.py --config configs/example/waymo_train_$scene.yaml mode trajectory
done
