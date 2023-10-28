echo ready to start first detection node
pause
@start python detect.py --port 9001
echo ready to start second detection node
pause
@start python detect.py --port 9002
echo ready to start  proxy
pause
@start python proxy.py
echo ready to player now...
pause
@python player.py
