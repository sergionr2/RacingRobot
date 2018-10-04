#!/bin/bash
# Launch an experiment using the docker cpu image

cmd_line="$@"

echo "Executing in the docker (cpu image):"
echo $cmd_line


docker run -it --rm --network host --ipc=host \
 --mount src=$(pwd),target=/tmp/racing_robot,type=bind araffin/racing-robot-cpu\
  bash -c "cd /tmp/racing_robot/ && $cmd_line"
