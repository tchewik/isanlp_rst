#!/bin/bash

script_dir=$(dirname $0)
docker build -t tchewik/isanlp_rst:1.1 $script_dir
