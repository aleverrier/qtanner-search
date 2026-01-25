# Scripts

## Publish best_codes for a group or order

```
bash scripts/publish_best_codes_group.sh --group C2xC2
bash scripts/publish_best_codes_group.sh --order 8
bash scripts/publish_best_codes_group.sh --group C2xC2 --verify
```

## Refine published codes by length (dist_m4ri)

```
bash scripts/refine_best_codes_length.sh --n 72 --trials-per-side 50000 --jobs 5
bash scripts/refine_best_codes_length.sh --n 72 --trials-per-side 50000 --dist-m4ri ~/tools/dist_m4ri
bash scripts/refine_best_codes_length.sh --n 72 --trials-per-side 50000 --verify
```
