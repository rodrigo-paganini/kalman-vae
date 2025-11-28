# Makefile for common dev tasks
# - `make latest-run` prints the latest run directory under `runs/`
# - `make tensorboard` starts tensorboard using the latest run directory

SHELL := /bin/bash

# Find latest run dir under runs/ (sorted by modification time)
LATEST_RUN := $(shell ls -1dt runs/* 2>/dev/null | head -n 1)

.PHONY: latest-run tensorboard

latest-run:
	@if [ -z "$(LATEST_RUN)" ]; then \
		echo "No run directories found under 'runs/'."; \
		exit 1; \
	fi; \
	echo "$(LATEST_RUN)"

board: latest-run
	@echo "Starting TensorBoard for: $(LATEST_RUN)"
	@tensorboard --logdir "$(LATEST_RUN)"
