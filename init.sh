#!/bin/bash

mkdir -p input
curl  -o input/mmdata.json --proto '=https' --tlsv1.2 -sSf "https://zenodo.org/record/6319684/files/mattermost.json"