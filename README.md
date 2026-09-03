# kirack — Attention (URMA) Policy for Humanoid Velocity Tracking

`master` 브랜치의 baseline(MLP actor-critic + PPO) 대비, **morphology-aware attention 정책**으로
동일 계열의 velocity tracking / standing 태스크를 학습하는 실험 브랜치입니다.

핵심 아이디어는 관측을 하나의 평탄한 벡터로 concat 하지 않고
**"관절 단위 / 발 단위 / 로봇 전역"** 텐서로 분해한 뒤, 학습 가능한 attention pooling 으로
가변 개수의 관절·발을 고정 크기 latent 로 집약하는 것입니다.
그 결과 **관절 수가 다른 로봇들을 하나의 정책 네트워크로 학습**할 수 있습니다
(KAPEX0 + Unitree G1 동시 학습).

> Baseline 은 `master` 브랜치를 참고하세요. 태스크 ID, 패키지 구조, PPO 설정이 모두 다릅니다.

---

## 1. master 와의 차이 요약

| | `master` (baseline) | `feat/attention` |
| --- | --- | --- |
| 정책 네트워크 | MLP `[512, 256, 128]` actor / critic | Attention pooling trunk + per-joint head |
| 관측 형태 | 단일 concat 벡터 (policy / critic 그룹, history 5) | 5개 flat 그룹 텐서 (`joint_desc`, `joint_obs`, `feet_desc`, `feet_obs`, `global_obs`) |
| 액션 헤드 | 전체 관절 한 번에 출력 | 관절별로 동일 헤드를 공유해 `(E, J)` 출력 |
| 지원 로봇 | KAPEX0 | KAPEX0, Unitree G1 23DOF, 두 로봇 동시 |
| 태스크 ID | `Kirack-Velocity-v0` | `Kapex-Standing-v0`, `G1-Standing-v0`, `Combined-Standing-v0` |
| 패키지 경로 | `kirack/tasks/locomotion/kapex0_vel_track` | `kirack/tasks/standing`, `kirack/tasks/environment/<robot>` |
| 분산 학습 | 단일 태스크 | `--tasks` 로 rank 별 서로 다른 morphology |
| 정책 std | 관절별 std | 스칼라 공유 std (`SharedStdGaussianDistribution`) |
| 배포 | — | `scripts/rsl_rl/export_onnx.py` (ONNX + TorchScript) |

## 2. 프로젝트 구조

```
kirack/
├── rsl_rl/                                   # git submodule (leggedrobotics/rsl_rl)
├── scripts/
│   ├── rsl_rl/
│   │   ├── train.py                          # --tasks 로 multi-morphology 분산 학습 지원
│   │   ├── play.py
│   │   └── export_onnx.py                    # ONNX / TorchScript 내보내기
│   ├── test_env.py                           # articulation data + 5개 attention 관측 덤프
│   └── test_exp_loader.py                    # 실험 YAML 오버라이드 스모크 테스트
└── source/kirack/kirack/
    ├── __init__.py                           # kirack_EXT_DIR / kirack_DATA_DIR
    └── tasks/
        ├── environment/                      # 로봇별 morphology 기술(description) 파이프라인
        │   ├── kapex/observations/observation.py   # (gitignore: 사내 asset)
        │   └── g1/
        │       ├── observations/observation.py     # URDF 파싱 → joint/feet/global 텐서
        │       └── utils/unitree.py, unitree_actuator.py
        └── standing/
            ├── __init__.py                   # gym.register (6개 태스크)
            ├── kirack_env_cfg.py             # KapexStanding(Play)EnvCfg
            ├── g1_standing_env_cfg.py        # G1Standing(Play)EnvCfg
            ├── combined_standing_env_cfg.py  # CombinedStanding(Play)EnvCfg (두 로봇 동시 스폰)
            ├── observations_combined.py      # Kapex + G1 관측 concat
            ├── agents/
            │   ├── attention_model.py        # AttentionTrunk / ActionMeanHead / ValueHead + Cfg
            │   ├── shared_std_gaussian.py    # 스칼라 공유 std Gaussian
            │   └── rsl_rl_ppo_cfg.py         # PPORunnerCfg (actor/critic + obs_groups)
            ├── mdp/                          # commands, curriculum, events, observations, rewards, terminations
            └── robot/                        # kapex.py, g1.py ArticulationCfg
```

## 3. 설치

1. [Isaac Lab](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html) 설치 (Isaac Sim 5.1 기준).
2. 저장소를 **서브모듈까지** 클론합니다. `rsl_rl` 은 submodule 입니다.

```bash
git clone --recurse-submodules <repo-url>
# 이미 클론했다면
git submodule update --init --recursive
```

3. Isaac Lab 파이썬으로 확장과 rsl_rl 을 editable 설치합니다.

```bash
python -m pip install -e source/kirack
python -m pip install -e rsl_rl        # >= 4.0.0 model API 필요
```

4. 확인:

```bash
python scripts/list_envs.py
```

> **Asset 안내** — KAPEX0 관련 파일(`tasks/environment/kapex/**`)은 `.gitignore` 로 제외되어 있습니다.
> 저장소만으로는 `Kapex-*` / `Combined-*` 태스크를 실행할 수 없고, 별도로 제공받아야 합니다.
> `G1-Standing-v0` 는 `tasks/environment/g1/robots/g1_description/g1_23dof.urdf` 가 필요합니다.

`train.py` 는 `python-dotenv` 로 `.env` 를 읽습니다 (예: `WANDB_API_KEY`).

## 4. 등록된 태스크

| Task ID | 로봇 | Env Cfg |
| --- | --- | --- |
| `Kapex-Standing-v0` / `-Play-v0` | KAPEX0 | `kirack_env_cfg:KapexStanding(Play)EnvCfg` |
| `G1-Standing-v0` / `-Play-v0` | Unitree G1 23DOF | `g1_standing_env_cfg:G1Standing(Play)EnvCfg` |
| `Combined-Standing-v0` / `-Play-v0` | KAPEX0 + G1 (한 env 에 동시 스폰) | `combined_standing_env_cfg:CombinedStanding(Play)EnvCfg` |

세 태스크 모두 동일한 `PPORunnerCfg` (attention actor/critic) 를 사용합니다.

## 5. 관측 구조 — URMA 5-tensor

기존처럼 하나의 벡터로 concat 하지 않고, env 가 **5개의 최상위 flat 그룹**을 그대로 내보냅니다
(rsl_rl 의 rollout storage 가 flat 구조를 요구하므로 그룹당 텀 1개).

| 그룹 | shape | 내용 |
| --- | --- | --- |
| `joint_desc` | `(E, J, 16)` | **정적 morphology.** root 프레임 기준 관절 앵커 위치(3), 관절 축(3), 자식 관절 수(1), 기본 자세(1), effort/velocity limit(2), stiffness/damping/armature/friction(4), 관절 위치 한계(2). reset 시 1회 계산 후 캐시. |
| `joint_obs` | `(E, J, 3)` | **동적 상태.** 관절별 `[pos_rel, vel_rel, last_action]`. |
| `feet_desc` | `(E, F, 3)` | 정적 발 위치 (root 프레임), 캐시. |
| `feet_obs` | `(E, F, 2)` | `[foot_z_in_root, foot_vz_in_root]`. |
| `global_obs` | `(E, 15)` | base 선속도(3), 각속도(3), projected gravity(3), 속도 명령(3), torso CoM 편차(3). |

`joint_desc` 는 URDF 를 직접 파싱해 만듭니다 (관절 축·트리 구조·부모/자식 링크).
로봇마다 `tasks/environment/<robot>/observations/observation.py` 가 이 5개 텐서를 생성하고,
**스키마(16 / 3 / 3 / 2 / 15)가 동일하므로 같은 정책이 두 로봇을 모두 소비할 수 있습니다.**

`Combined-Standing-v0` 는 요소 축으로 `[Kapex, G1]` 순서로 이어붙입니다
(`observations_combined.py` 기준 44 joints = 21 + 23, 4 feet, 30-dim global).

## 6. 정책 아키텍처 (`agents/attention_model.py`)

JAX 구현체 `one_policy_to_run_them_all` 을 PyTorch / rsl_rl `MLPModel` 인터페이스로 옮긴 것입니다.

### AttentionTrunk (actor / critic 공용 구조)

```
joint_desc (E,J,16) ──> Linear(64) ─ LayerNorm ─ ELU ─ Linear(M=64) ─ tanh ─┐
                                                                            ├─ softmax(m / T) over feature axis
joint_obs  (E,J,3)  ──> Linear(S=32) ─ ELU ──────────────────────────────── ┘   → weights (E,J,M)
                                          gated = weights ⊗ state_latent  (E,J,M,S)
                                          pooled = Σ_J flatten(M·S)        (E, 2048)

feet_desc / feet_obs ─ 동일 구조 (M=32, S=32) ────────────────────────────→ (E, 1024)

concat(joint_pool, foot_pool, global_obs)
  → Linear(512) ─ LayerNorm ─ ELU → Linear(256) ─ ELU → Linear(128)  = trunk latent
```

- 여기서 attention 은 transformer self-attention 이 아니라, **description 으로부터 만든 mask 를
  feature 축에 softmax 로 정규화해 각 요소의 state latent 를 게이팅**하는 방식입니다.
- 관절 축(J) 으로 합산하므로 **관절 개수에 독립적인 고정 크기 latent** 가 나옵니다. 이것이
  서로 다른 로봇을 같은 네트워크로 다룰 수 있는 이유입니다.
- softmax temperature 는 학습 가능 파라미터입니다 (`log(T - T_min)` 으로 저장, `T_min = 0.015`).
- trunk 는 orthogonal init (gain √2) 을 씁니다.

### ActionMeanHead (actor)

`[trunk latent(타일링), sg(joint_state_latent), desc_encoder(joint_desc)]` → `Linear(128) ─ LN ─ ELU ─ Linear(1)`
→ 관절마다 스칼라 평균을 뽑아 `(E, J)`. 마지막 층 gain 0.01, 출력은 ±10 클립.
`joint_state_latent` 은 stop-gradient 로 넣어 trunk 를 통해 두 번 학습되지 않게 합니다.

### ValueHead (critic)

trunk latent → `ELU` → `Linear(1)` → `(E, 1)`. actor 와 동일한 trunk 구조를 별도 인스턴스로 사용합니다.

### 정규화

`joint_desc` 는 effort/stiffness/armature 등 스케일이 5자리 이상 차이 나므로,
그룹별 `EmpiricalNormalization` 을 모델 내부에 두고 rollout 마다 갱신합니다.

### 출력 분포 — `SharedStdGaussianDistribution`

rsl_rl 기본 `GaussianDistribution` 은 `std_param` 을 `(output_dim,)` 로 잡습니다.
KAPEX(21) 과 G1(23) 을 rank 별로 나눠 학습하면 **파라미터 shape 이 rank 마다 달라져 NCCL all_reduce 가 실패**합니다.
따라서 std 를 학습 가능한 **스칼라 1개**로 두어 모든 rank 가 동일한 shape 을 갖게 했습니다.

### 내보내기

- `as_jit()` → `_TorchAttentionActorModel` (attention pool 을 인라인해 `torch.jit.script` 가능)
- `as_onnx()` → `_OnnxAttentionActorModel` (5개 named input, `n_joints` / `n_feet` dynamic axis)
- critic 의 JIT/ONNX 내보내기는 미구현입니다.

## 7. 실행

### 단일 로봇 학습

```bash
python scripts/rsl_rl/train.py \
    --task G1-Standing-v0 \
    --num_envs 4096 --headless --logger wandb
```

### Multi-morphology 분산 학습

rank 마다 다른 로봇 환경을 띄우고, 하나의 attention 정책을 공유해 gradient 를 동기화합니다.
`tasks[LOCAL_RANK % len(tasks)]` 로 태스크가 배정됩니다.

```bash
python -m torch.distributed.run --nproc_per_node=2 scripts/rsl_rl/train.py \
    --tasks Kapex-Standing-v0,G1-Standing-v0 \
    --distributed --num_envs 4096 --headless --logger wandb
```

한 시뮬레이션 안에 두 로봇을 같이 띄우는 방식도 가능합니다 (GPU 1장):

```bash
python scripts/rsl_rl/train.py --task Combined-Standing-v0 --num_envs 2048 --headless
```

### 재생

```bash
python scripts/rsl_rl/play.py \
    --task G1-Standing-Play-v0 \
    --load_run <run_name> --num_envs 16
```

morphology 에 독립적인 정책이므로 **G1 으로 학습한 체크포인트를 Kapex 환경에서 재생**하는
cross-morphology 평가가 가능합니다.

```bash
python scripts/rsl_rl/play.py --task Kapex-Standing-Play-v0 --load_run <g1_run_name> --num_envs 16
```

### ONNX / TorchScript 내보내기

```bash
python scripts/rsl_rl/export_onnx.py \
    --task Kapex-Standing-Play-v0 \
    --load_run <run_name>
# 기본 출력: <checkpoint_dir>/exported/{policy.onnx, policy.pt}
```

`--output_dir`, `--onnx_filename`, `--jit_filename`, `--skip_jit` 로 조정할 수 있습니다.
(`num_envs=1`, `headless` 강제)

### 환경 점검

```bash
python scripts/test_env.py --task G1-Standing-Play-v0 --num_envs 2 --no_terrain
python scripts/test_env.py --task Kapex-Standing-Play-v0 --steps 5
python scripts/test_exp_loader.py --headless --exp_cfg <path/to/exp.yaml>
```

`test_env.py` 는 `robot.data.*` 와 5개 attention 관측을 덤프합니다.
URDF 파싱, 관절 순서, body 이름 해석을 학습 전에 검증하는 용도입니다.

## 8. 태스크 정의

### 제어 주기 (Kapex / G1 공통)

| 항목 | 값 |
| --- | --- |
| 물리 dt | 0.005 s (200 Hz) |
| decimation | 4 → 정책 50 Hz |
| 에피소드 길이 | 20 s |
| 병렬 env | 4096 (`Combined` 는 `env_spacing=5.0`) |

### 액션

`JointPositionActionCfg` (전체 관절, `scale=0.25`, default offset).
`Combined` 는 `kapex_joint_pos`(21) + `g1_joint_pos`(23) 두 텀을 순서대로 두어,
정책 출력 44차원의 앞 21개가 Kapex, 뒤 23개가 G1 으로 흘러갑니다.

### 커맨드

`UniformLevelVelocityCommandCfg` — standing 위주라 시작 범위가 좁습니다.

| | 시작 | 커리큘럼 한계 |
| --- | --- | --- |
| `lin_vel_x` | (-0.1, 0.1) | (-0.5, 1.0) |
| `lin_vel_y` | (-0.1, 0.1) | (-0.5, 0.5) |
| `ang_vel_z` | (-0.2, 0.2) | (-0.2, 0.2) |

`rel_standing_envs=0.05`, 리샘플링 10 s.

### 보상

`track_lin_vel_xy`, `track_ang_vel_z`, `alive`, `base_height`(Kapex 0.91 m / G1 0.74 m),
`gait`, `feet_phase`, `feet_slide`, `feet_clearance`, `feet_to_close`, `undesired_contacts`,
그리고 정형화 항목 (`joint_vel`, `joint_acc`, `action_rate`, `energy`, `dof_pos_limits`,
`joint_deviation_arms/waists/legs`).
`Combined` 는 로봇 무관 항목은 한 번, 로봇별 항목은 `SceneEntityCfg` 를 바꿔 두 벌 정의합니다.

### 종료 조건

`time_out`, 지형 상대 `base_height < 0.5 m`, `bad_orientation > 0.8 rad`.
`Combined` 에서는 둘 중 하나만 실패해도 env 가 종료됩니다.

### 이벤트 / 커리큘럼

master 와 같은 도메인 랜덤화 세트(발 마찰, 링크 질량, torso CoM, armature, actuator gain, push)에
**`cache_joint_desc` / `cache_feet_desc` reset 이벤트**가 추가됩니다.
정적 description 텐서를 default pose 에서 1회 계산해 캐시하는 역할입니다.

커리큘럼: `terrain_levels_survival`, `lin_vel_cmd_levels`, `actuator_armature_range_levels`,
`actuator_gain_range_levels`, `push_robot_levels`, `torso_com_levels`.

## 9. PPO 설정

`source/kirack/kirack/tasks/standing/agents/rsl_rl_ppo_cfg.py`

```python
actor  = RslRlAttentionActorModelCfg()
critic = RslRlAttentionCriticModelCfg()
obs_groups = {
    "actor":  ["joint_desc", "joint_obs", "feet_desc", "feet_obs", "global_obs"],
    "critic": ["joint_desc", "joint_obs", "feet_desc", "feet_obs", "global_obs"],
}
```

| 항목 | 값 |
| --- | --- |
| `joint_mask_dim` / `foot_mask_dim` / `state_latent_dim` | 64 / 32 / 32 |
| `trunk_hidden` | `[512, 256, 128]` |
| `desc_hidden` / `mean_hidden` | 128 / 128 |
| `init_temperature` / `min_temperature` | 1.0 / 0.015 |
| `mean_abs_clip` | 10.0 |
| obs normalization | 그룹별 `EmpiricalNormalization` ON |
| `num_steps_per_env` | 24 |
| `num_learning_epochs` / `num_mini_batches` | 5 / 4 |
| learning rate | 1e-3, `adaptive` (desired KL 0.01) |
| `gamma` / `lam` | 0.99 / 0.95 |
| `clip_param` / `entropy_coef` | 0.2 / 0.01 |
| `experiment_name` | `kirack` |

actor 와 critic 은 같은 관측 그룹을 보므로, baseline 의 asymmetric actor-critic(critic 만
`height_scan`·`feet_air_time` 특권 관측)과 달리 **대칭 구조**입니다.

## 10. 실험 설정 오버라이드 (`--exp`)

master 와 동일하게 `--exp <path>` → `KIRACK_EXP_CFG` 환경 변수 → `kirack/utils/exp_loader.py` 경로로
YAML 오버라이드가 적용됩니다. 지원 키는 `agent`, `vel_command`, `vel_limit`, `rewards`,
`observations`, `events`, `terminations`, `curriculum`, `terrain`, `scene` 입니다.
값을 `null` 로 두면 해당 텀이 비활성화됩니다.

```yaml
agent:
  max_iterations: 30000
  algorithm:
    learning_rate: 5.0e-4
    entropy_coef: 0.005
rewards:
  feet_clearance: {weight: 0.5}
  gait: null
terrain:
  type: flat
```

동작 확인은 `scripts/test_exp_loader.py` 로 할 수 있습니다.

## 11. 코드 포맷

```bash
pip install pre-commit
pre-commit run --all-files
```

ruff 설정에서 `rsl_rl/` submodule 은 lint 대상에서 제외됩니다.

## 12. 참고

- URMA / one-policy-to-run-them-all: 하나의 정책으로 다양한 morphology 를 다루는 attention 아키텍처
- [Isaac Lab](https://isaac-sim.github.io/IsaacLab/) · [rsl_rl](https://github.com/leggedrobotics/rsl_rl)

---

Author: Sangyun Bae — Research Intern, KIST ARC LAB · <https://tkdyun.xyz>
