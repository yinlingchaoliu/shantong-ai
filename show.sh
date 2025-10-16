#!/bin/sh

error() {
  echo "\033[0;31m$1\033[0m"
}

port() {
  lsof -i:$1 | grep LISTEN 
}

