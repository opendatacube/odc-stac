#!/bin/bash

set -eu

create_pip_tree() {
    local src="${1:-.}"
    local dst="${2:-.}"
    dst=$(readlink -f "${dst}")

    for w in $(cd "$src" && find . -name "*.whl" | awk -F - '{sub("^./", ""); print $1}')
    do
        local base="${w//_/-}"
        local out="${dst}/${base}"
        mkdir -p "${out}"

        echo "${src}/${w}"* "-> ${out}/"
        echo "${src}/${base}"*tar.gz "-> ${out}/"
        cp "${src}/${w}"*.whl "${out}/"
        cp "${src}/${base}"*.tar.gz "${out}/"
    done
}

create_pip_tree "$@"
