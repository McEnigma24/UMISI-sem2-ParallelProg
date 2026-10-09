#!/bin/bash

clear
{
    if [[ -n "$(git status --porcelain)" ]]; then
    git add .
    git commit -m "fast"
    git push
    fi

    git pull --rebase
}
clear

