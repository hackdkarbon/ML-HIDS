echo ready to start first detection node.. 
@pause > nul
@start python detect.py --port 9001 
echo ready to start second detection node ...
@pause > nul
@start python detect.py --port 9002
echo ready to start  proxy ...
@pause > nul
@start python proxy.py
echo ready to player now...
@pause > nul
@python player.py
