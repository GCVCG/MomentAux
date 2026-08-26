# Checkpoint identity audit (machine-generated: python analysis/aggregate_identity.py --verify-dir results/identity)
#
# A checkpoint can load perfectly and be the WRONG NETWORK. Every probed cell
# on the cluster tree was EVALUATED against its own recorded accuracy, on both
# best.pt and last.pt. The per-shard reports this summarises are in
# results/identity/.

shard reports: 16 best, 16 last

==============================================================
cells with a best.pt result        : 2262
  not verifiable (all seeds ERR)   : 0
  VERIFIABLE                       : 2262
  at least one bad best.pt seed    : 79  (3.49%)
==============================================================
I1  band 2-10%   -> HIT
F-I1 (>25% => corpus unreliable)   -> dead
F-I3 (<0.5% => isolated accidents) -> dead

FAILURE MAGNITUDES (|recorded - evaluated|, accuracy points)
     27  0.5-1
     14  1-2
     17  2-5
     38  5-15
     21  >15
  -> 76/117 seed-level failures are >= 2 points, i.e. beyond any plausible decode drift

I2  grid-lane share of failures    : 79/79 (100%)   band >=60% -> HIT
I3  last.pt intact where best fails: 79/79 (100%)   band >=70% -> HIT

BLAST RADIUS: 107 released rows pair against a failing arm
    11 rows <- grid_food_r18_9ee7da_50pct
     7 rows <- grid_food_r18_9ee7da_25pct
     5 rows <- grid_mnet_food_none_50pct
     3 rows <- diaggrid_vit_food101_none_5pct
     3 rows <- diaggrid_vit_pathmnist_none_5pct
     2 rows <- diaggrid_c100_r18_cosinehead_2f17a9_5pct
     2 rows <- diaggrid_c100_r18_e400_06c094_1pct
     2 rows <- diaggrid_deit_c10_none_1pct
     2 rows <- diaggrid_swin_esat_none_2pct
     1 rows <- diaggrid_c100_r18_axmagnitudeL3_l10to00_cosinehead_d1f3e0_15pct
     1 rows <- diaggrid_c100_r18_axmagnitudeL3_l10to00_cosinehead_d1f3e0_25pct
     1 rows <- diaggrid_c100_r18_e100_4cd1ae_5pct
     1 rows <- diaggrid_c100_r18_e100_stmomentscatk11_kw03e54b_1201eb_1pct
     1 rows <- diaggrid_c100_r18_e100_stmomentscatk11_kw03e54b_1201eb_5pct,diaggrid_c100_r18_e100_4cd1ae_5pct
     1 rows <- diaggrid_c100_r18_e20_stmomentscatk11_d25252_5pct
