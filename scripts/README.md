# Scripts

## Publish best_codes for a group or order

```
bash scripts/publish_best_codes_group.sh --group C2xC2
bash scripts/publish_best_codes_group.sh --order 8
bash scripts/publish_best_codes_group.sh --group C2xC2 --verify
```

## Refine published codes by length (dist_m4ri)

```
./scripts/py scripts/refine_best_codes.py --n 72 --trials 50000
./scripts/py scripts/refine_best_codes.py --n 144 --trials 200000 --group C2xC6 --no-git --no-publish
```
