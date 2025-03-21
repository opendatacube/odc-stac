#!/bin/bash

main_env="$HOME/envs/main"

process_main_env() {
    declare -l -a opts
    local main_py="$main_env/bin/python"

    [ -d "$main_env" ] || {
        echo "Creating main virtual environment..."
        mkdir -p "$(dirname "$main_env")"
        uv venv "$main_env"
    }
    for req in .devcontainer/requirements*.txt; do
        opts+=(-r "$req")
    done
    source "$main_env"/bin/activate
    uv pip install "${opts[@]}"
}

process_apt() {
    local apt_file=".devcontainer/apt.txt"
    if [ -e ${apt_file} ]; then
        sudo apt-get -y update
        awk '{if ($0 ~ /^[[:space:]]*#/) next; sub(/#[^"]*$/, ""); print}' <"${apt_file}" |
            xargs sudo apt-get -y install
    fi
}

cd /workspace || exit 1
process_main_env
process_apt
