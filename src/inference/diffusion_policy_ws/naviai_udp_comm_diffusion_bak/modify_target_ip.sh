#!/bin/bash

SCRIPT_DIR=$(realpath $(dirname "$BASH_SOURCE"))

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <HOST_IP> <ORIN_IP>"
    exit 1
fi

HOST_IP="$1"
ORIN_IP="$2"

update_roslaunch_target_ip() {
    if [[ $# -lt 2 ]]; then
        echo "Usage: update_target_ip <NEW_IP> <LAUNCH_FILE>"
        return 1
    fi

    local NEW_IP="$1"
    local TARGET_LAUNCH="$2"

    if [[ ! -f "$TARGET_LAUNCH" ]]; then
        local EXAMPLE_FILE="${TARGET_LAUNCH}.example"
        if [[ -f "$EXAMPLE_FILE" ]]; then
            cp "$EXAMPLE_FILE" "$TARGET_LAUNCH"
            echo "Copied example file to create missing launch file: $TARGET_LAUNCH"
        else
            echo "Error: Launch file '$TARGET_LAUNCH' and example file '$EXAMPLE_FILE' both do not exist."
            return 1
        fi
    fi

    sed -i "s#<param name=\"target_ip\" value=\"[^\"]*\" />#<param name=\"target_ip\" value=\"${NEW_IP}\" />#g" "$TARGET_LAUNCH"

    echo "Replaced target_ip with ${NEW_IP} in ${TARGET_LAUNCH}"
}

update_sh_ip() {
    if [[ $# -lt 2 ]]; then
        echo "Usage: update_sh_ip <NEW_IP> <TARGET_SH>"
        return 1
    fi

    local NEW_IP="$1"
    local TARGET_PATH="$2"

    if [[ ! -f "$TARGET_PATH" ]]; then
        echo "Error: Shell script '$TARGET_PATH' does not exist."
        return 1
    fi

    sed -i "s#^IP_ADDRESS=\"[0-9\.]*\"#IP_ADDRESS=\"${NEW_IP}\"#g" "$TARGET_PATH"

    echo "Updated IP_ADDRESS to ${NEW_IP} in ${TARGET_PATH}"
}
