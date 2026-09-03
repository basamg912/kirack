from isaaclab.assets import ArticulationCfg
from kirack.assets.robot.kapex0 import KAPEX0_CFG
KAPEX_CFG = KAPEX0_CFG

'''
-------------- ROBOT DESCRIPTION
----- joint - init_state
Link name : J 만 빼면 됨

# ── LEGS (LLJ / RLJ) ───────────────────────────────────────── 14개
0  'LLJ1'   1  'RLJ1'
3  'LLJ2'   4  'RLJ2'
6  'LLJ3'   7  'RLJ3'
9  'LLJ4'   10 'RLJ4'
14 'LLJ5'   15 'RLJ5'
19 'LLJ6'   20 'RLJ6'
23 'LLJ7'   24 'RLJ7'

# ── WAIST (WLJ) ────────────────────────────────────────────── 3개
2  'WLJ1'   5  'WLJ2'   8  'WLJ3' -- WLJ3 : 몸통 가장 큰 비중 차지

# ── HEAD (HLJ) ─────────────────────────────────────────────── 2개
11 'HLJ1'   16 'HLJ2'

# ── ARMS (LAJ / RAJ) ───────────────────────────────────────── 14개
12 'LAJ1'   13 'RAJ1'
17 'LAJ2'   18 'RAJ2'
21 'LAJ3'   22 'RAJ3'
25 'LAJ4'   26 'RAJ4'
27 'LAJ5'   28 'RAJ5'
29 'LAJ6'   30 'RAJ6'
31 'LAJ7'   32 'RAJ7'

팔 제외 : 33 dof (0~33)

# ── LEFT HAND (LHJ) ────────────────────────────────────────── 20개
33 'LHJ_index0'   34 'LHJ_little0'   35 'LHJ_middle0'   36 'LHJ_ring0'   37 'LHJ_thumb0'
43 'LHJ_index1'   44 'LHJ_little1'   45 'LHJ_middle1'   46 'LHJ_ring1'   47 'LHJ_thumb1'
53 'LHJ_index2'   54 'LHJ_little2'   55 'LHJ_middle2'   56 'LHJ_ring2'   57 'LHJ_thumb2'
63 'LHJ_index3'   64 'LHJ_middle3'   65 'LHJ_ring3'     66 'LHJ_thumb3'
71 'LHJ_thumb4'

# ── RIGHT HAND (RHJ) ───────────────────────────────────────── 20개
38 'RHJ_index0'   39 'RHJ_little0'   40 'RHJ_middle0'   41 'RHJ_ring0'   42 'RHJ_thumb0'
48 'RHJ_index1'   49 'RHJ_little1'   50 'RHJ_middle1'   51 'RHJ_ring1'   52 'RHJ_thumb1'
58 'RHJ_index2'   59 'RHJ_little2'   60 'RHJ_middle2'   61 'RHJ_ring2'   62 'RHJ_thumb2'
67 'RHJ_index3'   68 'RHJ_middle3'   69 'RHJ_ring3'     70 'RHJ_thumb3'
72 'RHJ_thumb4'
'''
