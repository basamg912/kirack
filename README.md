# kirack

KIST ARC LAB 휴머노이드 **KAPEX** 의 velocity tracking 정책을 Isaac Lab 위에서 학습시키는 프로젝트입니다.

> [!caution]
> **Kapex 의 URDF, XML, 모델 파일 등 로봇 관련 파일은 Repo 에 포함되어 있지 않습니다.**

`master` 브랜치는 **baseline** 입니다. 표준적인 MLP actor-critic + PPO 구성입니다.
`feat/attention` 브랜치는 attention 기반 정책을 학습하는 브랜치입니다. (URMA style - One-policy-to-run-them-all)

- 시뮬레이터: Isaac Sim 5.1 / Isaac Lab (manager-based RL env)
- RL: `rsl_rl` PPO (`RslRlPpoActorCriticCfg`)
- 로깅: TensorBoard / Weights & Biases

---

## 1. 프로젝트 구조

```
kirack/
├── scripts/
│   ├── list_envs.py                  # 등록된 gym 태스크 목록
│   ├── zero_agent.py / random_agent.py
│   ├── test_env.py                   # 환경 sanity check
│   └── rsl_rl/
│       ├── train.py                  # 학습 진입점
│       ├── play.py                   # 체크포인트 재생 / 비디오 녹화
│       └── cli_args.py
└── source/kirack/kirack/
    ├── __init__.py                   # gym.register (Kirack-Velocity-v0 / -Play-v0)
    ├── kirack_env.py                 # KirackEnvCfg / KirackPlayEnvCfg (robot 주입 + viewer + exp 오버라이드)
    ├── assets/robot/                 # KAPEX (USD/설명 파일은 .gitignore)
    ├── tasks/locomotion/kapex0_vel_track/
    │   ├── kapex0_vel_track_env_cfg.py   # scene / obs / action / reward / event / curriculum 본체
    │   ├── agents/rsl_rl_ppo_cfg.py      # PPORunnerCfg
    │   └── config/                       # 커스텀 MDP 텀
    │       ├── commands.py               # UniformLevelVelocityCommandCfg
    │       ├── curriculum.py             # terrain / vel / actuator / push / com 레벨링
    │       ├── event.py                  # 도메인 랜덤화 텀
    │       ├── observation.py            # torso_com, obs_feet_air_time
    │       ├── reward.py                 # gait, feet clearance, energy, mirror 등
    │       └── terminations.py
    └── utils/
        ├── exp_loader.py             # YAML 실험 설정 오버라이드
        └── wandb_video.py
```

## 2. 설치

1. [Isaac Lab 설치 가이드](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html)에 따라 Isaac Lab 을 먼저 설치합니다. (conda 또는 uv 권장)
2. 이 저장소를 `IsaacLab` 디렉터리 **밖에** 클론합니다.
3. Isaac Lab 이 설치된 파이썬 인터프리터로 확장을 editable 모드로 설치합니다.

```bash
# Isaac Lab 이 venv/conda 에 없으면 'python' 대신 'PATH_TO_isaaclab.sh -p' 사용
python -m pip install -e source/kirack
```

설치 확인:

```bash
python scripts/list_envs.py
```

`Kirack-Velocity-v0`, `Kirack-Velocity-Play-v0` 두 태스크가 보이면 정상입니다.

> `assets/robot/kapex0.py` 와 KAPEX0 description(USD/URDF) 은 `.gitignore` 로 제외되어 있습니다.
> 저장소만 클론하면 로봇 asset 이 없으므로, 별도로 제공받아 `source/kirack/kirack/assets/robot/` 아래에 두어야 합니다.

## 3. 등록된 태스크

| Task ID | Env Cfg | 용도 |
| --- | --- | --- |
| `Kirack-Velocity-v0` | `KirackEnvCfg` | 학습 (4096 envs, curriculum ON, rough terrain) |
| `Kirack-Velocity-Play-v0` | `KirackPlayEnvCfg` | 재생/평가 (world 시점, curriculum OFF) |

## 4. 실행

### 학습

```bash
python scripts/rsl_rl/train.py \
    --task Kirack-Velocity-v0 \
    --num_envs 4096 --headless \
    --logger wandb --video --video_interval 10000
```

### 이어서 학습 (fine-tune)

```bash
python scripts/rsl_rl/train.py \
    --task Kirack-Velocity-v0 --num_envs 4096 --headless \
    --resume --load_run <run_name> --checkpoint model_4000.pt
```

### 재생 / 비디오

```bash
python scripts/rsl_rl/play.py \
    --task Kirack-Velocity-Play-v0 \
    --num_envs 5 --video --video_length 300 --headless \
    --checkpoint logs/rsl_rl/Kapex-Vel-Tracking/<run>/model_*.pt
```

### 환경 점검용 더미 에이전트

```bash
python scripts/zero_agent.py   --task Kirack-Velocity-v0
python scripts/random_agent.py --task Kirack-Velocity-v0
python scripts/test_env.py     --task Kirack-Velocity-v0
```

## 5. 태스크 정의

### 제어 주기

| 항목 | 값 |
| --- | --- |
| 물리 dt | 0.005 s (200 Hz) |
| decimation | 4 |
| 정책 dt | 0.02 s (50 Hz) |
| 에피소드 길이 | 8 s |

### 커맨드

`UniformLevelVelocityCommandCfg` — `UniformVelocityCommandCfg` 에 `limit_ranges` 를 추가한 커스텀 커맨드입니다.
커리큘럼이 성공률에 따라 `ranges` 를 `limit_ranges` 까지 점진적으로 넓힙니다.

| | 시작 범위 | 최종 한계 |
| --- | --- | --- |
| `lin_vel_x` | (-0.1, 0.5) | (-0.5, 1.2) |
| `lin_vel_y` | (-0.1, 0.1) | (-0.5, 0.5) |
| `ang_vel_z` | (-0.2, 0.2) | (-0.2, 0.2) |

`rel_standing_envs=0.3` — 30% 환경은 정지 명령을 받아 locomotion 과 standing 을 함께 학습합니다.
리샘플링 주기는 4~5 s 입니다.

### 관측 (asymmetric actor-critic)

policy 그룹에는 노이즈를 주입하고, critic 그룹은 노이즈 없이 특권 정보를 추가로 받습니다.
두 그룹 모두 `history_length = 5` 입니다.

| 항목 | policy | critic |
| --- | --- | --- |
| `base_lin_vel` / `base_ang_vel` | ✓ (노이즈) | ✓ |
| `projected_gravity` | ✓ (노이즈) | ✓ |
| `velocity_commands` | ✓ | ✓ |
| `joint_pos_rel` / `joint_vel_rel` | ✓ (노이즈) | ✓ |
| `last_action` | ✓ | ✓ |
| `dif_torso_com` (WL3 CoM 편차) | ✓ | ✓ |
| `height_scan` (1.6 × 1.0 grid) | — | ✓ |
| `obs_feet_air_time` | — | ✓ |

### Action

`JointPositionActionCfg` — 전체 관절, `scale=0.25`, default pose offset 사용.

### 보상 (주요 항목)

- **추종**: `track_lin_vel_xy`, `track_ang_vel_z`
- **자세/생존**: `base_height` (목표 0.91 m), `terminated` (-200), `base_linear_velocity`, `base_angular_velocity`
- **매끄러움**: `joint_vel`, `joint_acc`, `action_rate`, `energy`, `dof_pos_limits`
- **자세 정형화**: `joint_deviation_arms`, `joint_deviation_waists`, `joint_position_penalty`
- **보행 품질**: `gait` (위상 기반 접촉 보상), `feet_slide`, `feet_clearance`, `undesired_contacts`
- **정지 상태**: `stand_still`, `feet_contact_without_cmd`

### 종료 조건

`time_out`, `base_height < 0.5 m` (지형 상대), `bad_orientation > 0.8 rad`.

### 지형

`TerrainGeneratorCfg` (9 level × 21 col, 3 × 3 m 타일):
`slope` 40% (pyramid, 0.0~0.5) / `boxes` 30% (grid, 높이 0~0.05 m) / `flat` 30%.
`max_init_terrain_level=0` 이라 초기 학습은 가장 쉬운 지형에서 시작합니다.

### 도메인 랜덤화 (Events)

- **startup**: 발 마찰/반발 계수, 동체(WL3) 질량 ±1 kg, 전체 링크 질량 ×0.95~1.05
- **reset**: 동체 CoM 오프셋, 관절 armature 스케일, actuator stiffness/damping, base/joint 리셋, 관절 위치 노이즈
- **interval**: `push_robot` 외란

### 커리큘럼

`terrain_levels_survival`, `lin_vel_cmd_levels`, `actuator_armature_range_levels`,
`actuator_gain_range_levels`, `push_robot_levels`, `torso_com_levels` —
학습 진행도에 따라 지형 난이도, 속도 명령 범위, 액추에이터 파라미터 랜덤화 폭, 외란 크기를 함께 키웁니다.

## 6. PPO 설정

`source/kirack/kirack/tasks/locomotion/kapex0_vel_track/agents/rsl_rl_ppo_cfg.py`

| 항목 | 값 |
| --- | --- |
| actor / critic hidden | `[512, 256, 128]`, ELU |
| obs normalization | actor / critic 모두 ON |
| `num_steps_per_env` | 24 |
| `num_learning_epochs` / `num_mini_batches` | 5 / 4 |
| learning rate | 1e-3, `adaptive` (desired KL 0.01) |
| `gamma` / `lam` | 0.99 / 0.95 |
| `clip_param` / `entropy_coef` | 0.2 / 0.005 |
| `max_iterations` | 50000 |
| experiment / wandb project | `Kapex-Vel-Tracking` / `Kirack` |

## 7. 실험 설정 오버라이드 (`--exp`)

`--exp <path/to/exp.yaml>` 로 코드 수정 없이 환경/에이전트 설정을 덮어쓸 수 있습니다.
경로는 `KIRACK_EXP_CFG` 환경 변수로 전달되고 `kirack/utils/exp_loader.py` 가 적용합니다.

```yaml
# exp/example.yaml
agent:
  max_iterations: 30000
  policy:
    init_noise_std: 0.8
  algorithm:
    learning_rate: 5.0e-4
    schedule: fixed

vel_command:                 # 시작 커맨드 범위
  lin_vel_x: [0.0, 0.8]
  lin_vel_y: [-0.2, 0.2]
  ang_vel_z: [-0.3, 0.3]
vel_limit:                   # 커리큘럼 상한
  lin_vel_x: [-0.5, 1.5]
  lin_vel_y: [-0.5, 0.5]
  ang_vel_z: [-0.5, 0.5]

rewards:
  feet_clearance: {weight: 0.5}
  stand_still: null          # null 이면 해당 텀 비활성화

observations:
  policy:
    enable_corruption: false
    joint_vel_rel: {scale: 0.1, clip: [-1.0, 1.0]}

events:
  push_robot: {params: {velocity_range: {x: [-0.5, 0.5]}}}
terminations:
  bad_orientation: {params: {limit_angle: 1.0}}
curriculum:
  terrain_levels: null

terrain:
  type: flat                 # flat | rough (rough 는 rough_ratio 지정 가능)
scene:
  env_spacing: 3.0
```

지원 키: `agent`, `vel_command`, `vel_limit`, `rewards`, `observations`, `events`,
`terminations`, `curriculum`, `terrain`, `scene`.
각 섹션에서 값을 `null` 로 두면 해당 텀이 비활성화됩니다.

## 8. 코드 포맷

```bash
pip install pre-commit
pre-commit run --all-files
```

---

Author: Sangyun Bae — Research Intern, KIST ARC LAB · <https://tkdyun.xyz>
