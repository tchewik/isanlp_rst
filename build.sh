#!/bin/bash

script_dir=$(dirname $0)
docker build -t tchewik/isanlp_rst $script_dir
