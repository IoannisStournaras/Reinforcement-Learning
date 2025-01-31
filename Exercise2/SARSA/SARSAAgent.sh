#!/bin/bash

# Number of defense agents must be added by one to take into account of goalkeeper
# Cannot run an environment where defending agents exist but none are playing
# goalkeeper

./../../../bin/HFO --defense-agents=2 --offense-agents=1 --offense-on-ball 11 --trials 5000 --headless --deterministic --discrete=True --frames-per-trial 2000 --untouched-time 2000 >/dev/null 2>&1 &
sleep 5
./DiscreteHFO/Initiator.py --numTrials=5000 --numPlayingDefenseNPCs=1 --numAgents=1  >/dev/null 2>&1 &
echo "Environment Initialized"
# Sleep is needed to make sure doesn't get connected too soon, as unum 1 (goalie)

sleep 5
./SARSABase.py --numOpponents=1 --numEpisodes=5000 &
echo "Attacker Controller Initialized" 

sleep 5
./DiscreteHFO/Goalkeeper.py --numEpisodes=5000  >/dev/null 2>&1 &
echo "Goalkeeper Initialized"

sleep 5
./DiscreteHFO/DiscretizedDefendingPlayer.py --numEpisodes=5000 --id=1 >/dev/null 2>&1 &
echo "Defending Player Initialized"

sleep 5
# The magic line
#   $$ holds the PID for this script
#   Negation means kill by process group id instead of PID
trap "kill -TERM -$$" SIGINT
wait
