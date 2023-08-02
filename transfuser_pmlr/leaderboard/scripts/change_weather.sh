export CARLA_ROOT=/home/mohammad/Mohammad_ws/autonomous_driving/e2etransfuser/carla
export WORK_DIR=/home/mohammad/Mohammad_ws/autonomous_driving/e2etransfuser/transfuser_pmlr

export CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
export SCENARIO_RUNNER_ROOT=${WORK_DIR}/scenario_runner
export LEADERBOARD_ROOT=${WORK_DIR}/leaderboard
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/":"${SCENARIO_RUNNER_ROOT}":"${LEADERBOARD_ROOT}":${PYTHONPATH}

export SCENARIOS=/home/mohammad/Mohammad_ws/autonomous_driving/e2etransfuser/leaderboard/data/scenarios/no_scenarios.json
export ROUTES=/home/mohammad/Mohammad_ws/autonomous_driving/e2etransfuser/leaderboard/data/all_routes/routes_town05_long.xml
export REPETITIONS=1
export CHALLENGE_TRACK_CODENAME=MAP
export CHECKPOINT_ENDPOINT=${WORK_DIR}/results/eval_result_all_weather.json
export SAVE_PATH=${WORK_DIR}/results/all_weather_test
export TEAM_AGENT=/home/mohammad/Mohammad_ws/autonomous_driving/e2etransfuser/transfuser_pmlr/team_code_autopilot/x13_agent2.py
export DEBUG_CHALLENGE=0
export RESUME=1
#export DATAGEN=1

python3 ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator_local.py \
--scenarios=${SCENARIOS}  \
--routes=${ROUTES} \
--repetitions=${REPETITIONS} \
--track=${CHALLENGE_TRACK_CODENAME} \
--checkpoint=${CHECKPOINT_ENDPOINT} \
--agent=${TEAM_AGENT} \
--agent-config=${TEAM_CONFIG} \
--debug=${DEBUG_CHALLENGE} \
--resume=${RESUME}
