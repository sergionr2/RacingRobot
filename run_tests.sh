#!/bin/bash
pytest -v --cov-config .coveragerc --cov-report html --cov-report term --cov=. tests/
